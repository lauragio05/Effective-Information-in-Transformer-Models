"""
biased_ioi_dataset_extra.py

A biased variant of the IOI dataset with an extra intermediate sentence
inserted between the introductory clause and the giving clause.

Example (Mary = IO):
    "When Mary and John went to the store, Mary had a good day. John gave a drink to Mary"
Example (John = IO):
    "When John and Mary went to the store, John had a good day. Mary gave a drink to John"

The intermediate sentence always names the IO:
    [IO] had a good day.
    [IO] was enjoying the situation.
    [IO] was tired.
    [IO] enjoyed being with a friend.
    [IO] was an enthusiast person.

Dataset composition (controlled by the `bias_ratio` parameter, default 0.7):
  - 70 % of prompts: Mary = IO (answer), John = S (giver)
  - 30 % of prompts: John = IO (answer), Mary = S (giver)

Both groups use the same family of templates with the [MIDDLE] slot filled
by one of the five intermediate-sentence templates above.

Switch parameter
----------------
switch_perc  (float, 0–1)
    Fraction of prompts in which the two names are swapped in the *first clause*
    only, i.e.
        normal : "When [IO] and [S] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]"
        switched: "When [S]  and [IO] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]"
    The IO/S labels are NOT changed — only the surface word order in the
    introductory clause flips.

Interface
---------
BiasedIOIDataset mirrors IOIDataset so it can be dropped in wherever IOIDataset
is used:
    .sentences        – list[str]
    .ioi_prompts      – list[dict]  keys: "text", "IO", "S", "TEMPLATE_IDX",
                                         "MIDDLE_IDX", "[PLACE]", "[OBJECT]",
                                         "switched"
    .toks             – LongTensor  [N, seq_len]
    .word_idx         – dict mapping semantic role → LongTensor of token indices
    .io_tokenIDs      – list[int]
    .s_tokenIDs       – list[int]
    .N                – int
    .max_len          – int
    .tokenized_prompts– list[str]  (pipe-separated token strings)
"""

import copy
import random
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Shared vocab (re-uses the same places / objects as the original ioi_dataset)
# ---------------------------------------------------------------------------
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

# ---------------------------------------------------------------------------
# Default names used in the biased dataset.
# IMPORTANT: both names must tokenize to a SINGLE GPT-2 token (as " Name").
# ---------------------------------------------------------------------------
FIXED_IO_NAME = "Mary"   # default majority IO name
FIXED_S_NAME  = "John"   # default majority S  name

# ---------------------------------------------------------------------------
# Intermediate-sentence templates.
# [IO] is replaced with the IO name at generation time.
# Each string ends with a period so the giving clause starts a new sentence.
# ---------------------------------------------------------------------------
MIDDLE_TEMPLATES = [
    "had a good day.",
    "was enjoying the situation.",
    "was tired.",
    "enjoyed being with a friend.",
    "was an enthusiast person.",
]

# ---------------------------------------------------------------------------
# Templates (intro clause + [MIDDLE] placeholder + giving clause)
# Full form:
#   "When [IO] and [S] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]"
# [MIDDLE] is replaced with one of the MIDDLE_TEMPLATES strings (which
# include the trailing period), so the result is two natural sentences.
# ---------------------------------------------------------------------------
BIASED_TEMPLATES = [
    "When [IO] and [S] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "When [IO] and [S] arrived at the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "After [IO] and [S] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "While [IO] and [S] were at the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "When [IO] and [S] were at the [PLACE], [IO] [MIDDLE] [S] decided to give a [OBJECT] to [IO]",
    "After [IO] and [S] arrived at the [PLACE], [IO] [MIDDLE] [S] decided to give a [OBJECT] to [IO]",
    "When [IO] and [S] visited the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "Once [IO] and [S] reached the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
]

# Switched form: first-clause order is flipped ([S] appears before [IO]).
# The intermediate sentence and giving clause stay identical.
BIASED_TEMPLATES_SWITCHED = [
    "When [S] and [IO] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "When [S] and [IO] arrived at the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "After [S] and [IO] went to the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "While [S] and [IO] were at the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "When [S] and [IO] were at the [PLACE], [IO] [MIDDLE] [S] decided to give a [OBJECT] to [IO]",
    "After [S] and [IO] arrived at the [PLACE], [IO] [MIDDLE] [S] decided to give a [OBJECT] to [IO]",
    "When [S] and [IO] visited the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
    "Once [S] and [IO] reached the [PLACE], [IO] [MIDDLE] [S] gave a [OBJECT] to [IO]",
]

assert len(BIASED_TEMPLATES) == len(BIASED_TEMPLATES_SWITCHED), (
    "Template lists must be the same length"
)


# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------

def gen_biased_prompts(
    N: int,
    switch_perc: float = 0.0,
    bias_ratio: float = 0.7,
    seed: Optional[int] = None,
    io_name: str = FIXED_IO_NAME,
    s_name:  str = FIXED_S_NAME,
) -> list:
    """Generate a list of biased IOI prompt dicts with an extra intermediate sentence.

    Parameters
    ----------
    N : int
        Total number of prompts to generate.
    switch_perc : float
        Fraction of prompts where the name order in the first clause is
        switched (S before IO) while IO/S labels remain unchanged.
    bias_ratio : float
        Fraction of prompts where Mary=IO / John=S (default 0.70).
        The remainder use John=IO / Mary=S.
    seed : int, optional
        Random seed for reproducibility.
    io_name : str
        Name used as the majority IO (default "Mary").
    s_name : str
        Name used as the majority S (default "John").

    Returns
    -------
    list of dict
        Each dict has keys: "text", "IO", "S", "TEMPLATE_IDX", "MIDDLE_IDX",
        "[PLACE]", "[OBJECT]", "switched".
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_majority = round(N * bias_ratio)
    n_minority = N - n_majority

    prompts = []

    for _io, _s, count in [
        (io_name, s_name, n_majority),   # majority group: io_name = IO
        (s_name, io_name, n_minority),   # minority group: s_name  = IO
    ]:
        for _ in range(count):
            place      = random.choice(PLACES)
            obj        = random.choice(OBJECTS)
            tmpl_idx   = random.randrange(len(BIASED_TEMPLATES))
            middle_idx = random.randrange(len(MIDDLE_TEMPLATES))

            switched = random.random() < switch_perc
            template = (BIASED_TEMPLATES_SWITCHED if switched
                        else BIASED_TEMPLATES)[tmpl_idx]

            middle_text = MIDDLE_TEMPLATES[middle_idx]

            text = (
                template
                .replace("[IO]",     _io)
                .replace("[S]",      _s)
                .replace("[PLACE]",  place)
                .replace("[OBJECT]", obj)
                .replace("[MIDDLE]", middle_text)
            )

            prompts.append({
                "text":         text,
                "IO":           _io,
                "S":            _s,
                "TEMPLATE_IDX": tmpl_idx,
                "MIDDLE_IDX":   middle_idx,
                "[PLACE]":      place,
                "[OBJECT]":     obj,
                "switched":     switched,
            })

    random.shuffle(prompts)
    return prompts


# ---------------------------------------------------------------------------
# Token-index helpers  (mirrors ioi_dataset.py conventions)
# ---------------------------------------------------------------------------

def _get_name_idxs(prompts, tokenizer, idx_types=("IO", "S", "S2"), prepend_bos=False):
    """Return token-index tensors for each requested name role.

    Note: S2 (second occurrence of S) is supported but optional.
    """
    name_idx_dict = {t: [] for t in idx_types}

    for prompt in prompts:
        t = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(t[:-1]))
        for idx_type in idx_types:
            role = idx_type.rstrip("2")          # "S2" → "S"
            tok  = tokenizer.tokenize(" " + prompt[role])[0]
            if "2" in idx_type:
                # last occurrence
                idx = len(toks) - toks[::-1].index(tok) - 1
            else:
                idx = toks.index(tok)
            name_idx_dict[idx_type].append(idx)

    return [
        int(prepend_bos) + torch.tensor(name_idx_dict[t])
        for t in idx_types
    ]


def _get_end_idxs(toks_tensor, tokenizer, name_tok_len=1, prepend_bos=False):
    """Return index of the last real (non-pad) token minus name_tok_len."""
    pad_id = tokenizer.pad_token_id
    relevant_idx = int(prepend_bos)
    end_idxs_raw = []

    for i in range(toks_tensor.shape[0]):
        if pad_id not in toks_tensor[i][1:]:
            end_idxs_raw.append(toks_tensor.shape[1])
        else:
            nonzers = (toks_tensor[i] == pad_id).nonzero()
            nonzers = nonzers[relevant_idx][0].item()
            end_idxs_raw.append(nonzers)

    end_idxs = torch.tensor(end_idxs_raw) - 1 - name_tok_len
    return end_idxs


def _build_word_idx(ioi_prompts, tokenizer, prepend_bos=False, toks=None):
    """Build the word_idx dict (same schema as IOIDataset.word_idx)."""
    idx_types = ["IO", "S", "S2"]
    IO_idxs, S_idxs, S2_idxs = _get_name_idxs(
        ioi_prompts, tokenizer, idx_types=idx_types, prepend_bos=prepend_bos
    )
    end_idxs = _get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=prepend_bos)

    return {
        "IO":     IO_idxs,
        "IO-1":   IO_idxs - 1,
        "IO+1":   IO_idxs + 1,
        "S":      S_idxs,
        "S-1":    S_idxs - 1,
        "S+1":    S_idxs + 1,
        "S2":     S2_idxs,
        "end":    end_idxs,
        "starts": torch.zeros_like(end_idxs),
    }


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class BiasedIOIDataset:
    """Biased IOI dataset with an extra intermediate sentence naming the IO.

    Each prompt follows the pattern:
        "[INTRO_CLAUSE], [IO] [MIDDLE_SENTENCE] [S] gave a [OBJECT] to [IO]"

    e.g. "When Mary and John went to the store, Mary had a good day.
          John gave a drink to Mary"

    Parameters
    ----------
    N : int
        Number of prompts.
    switch_perc : float
        Fraction of prompts where the introductory-clause name order is
        swapped while IO/S labels remain unchanged.
    bias_ratio : float
        Fraction of prompts where Mary=IO (default 0.70).
    tokenizer : transformers tokenizer, optional
        Defaults to GPT-2.
    prompts : list of dict, optional
        Supply pre-built prompt dicts instead of generating new ones.
    prepend_bos : bool
        Whether to prepend a BOS token.
    seed : int, optional
        Random seed.
    io_name : str
        Name used as the majority IO (default "Mary").
    s_name : str
        Name used as the majority S (default "John").
    """

    switch_perc: float = 0.0

    def __init__(
        self,
        N: int = 500,
        switch_perc: float = 0.0,
        bias_ratio: float = 0.7,
        tokenizer=None,
        prompts=None,
        prepend_bos: bool = False,
        seed: Optional[int] = None,
        io_name: str = FIXED_IO_NAME,
        s_name:  str = FIXED_S_NAME,
    ):
        self.switch_perc = switch_perc
        self.bias_ratio  = bias_ratio
        self.prepend_bos = prepend_bos
        self.io_name     = io_name
        self.s_name      = s_name

        # ── Tokenizer ────────────────────────────────────────────────────────
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # ── Prompts ──────────────────────────────────────────────────────────
        if prompts is not None:
            assert len(prompts) == N, (
                f"len(prompts)={len(prompts)} does not match N={N}"
            )
            self.ioi_prompts = prompts
        else:
            self.ioi_prompts = gen_biased_prompts(
                N=N,
                switch_perc=switch_perc,
                bias_ratio=bias_ratio,
                seed=seed,
                io_name=io_name,
                s_name=s_name,
            )

        self.N = len(self.ioi_prompts)

        # ── Sentences (list of strings) ───────────────────────────────────────
        self.sentences = [p["text"] for p in self.ioi_prompts]

        # ── Tokenize ─────────────────────────────────────────────────────────
        bos = self.tokenizer.bos_token if prepend_bos else ""
        texts = [bos + p["text"] for p in self.ioi_prompts]
        self.toks = torch.tensor(
            self.tokenizer(texts, padding=True).input_ids,
            dtype=torch.int,
        )

        # ── Word indices ──────────────────────────────────────────────────────
        self.word_idx = _build_word_idx(
            self.ioi_prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )

        # ── Token IDs for IO and S ────────────────────────────────────────────
        self.io_tokenIDs = [
            self.tokenizer.encode(" " + p["IO"])[0] for p in self.ioi_prompts
        ]
        self.s_tokenIDs = [
            self.tokenizer.encode(" " + p["S"])[0] for p in self.ioi_prompts
        ]

        # ── Misc ──────────────────────────────────────────────────────────────
        self.max_len = max(
            len(self.tokenizer(p["text"]).input_ids) for p in self.ioi_prompts
        )

        self.tokenized_prompts = [
            "|".join(self.tokenizer.decode(tok) for tok in self.toks[i])
            for i in range(self.N)
        ]

        # Convenience: template-group index (mirrors IOIDataset.groups)
        all_ids    = [p["TEMPLATE_IDX"] for p in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = [
            np.where(all_ids_ar == tid)[0] for tid in sorted(set(all_ids))
        ]
        small = [len(g) for g in self.groups if len(g) < 5]
        if small:
            warnings.warn(f"Some template groups have fewer than 5 prompts: {small}")

        # Majority / minority masks (useful for slicing)
        self.majority_mask = torch.tensor(
            [p["IO"] == self.io_name for p in self.ioi_prompts]
        )
        self.minority_mask = ~self.majority_mask
        self.switched_mask = torch.tensor(
            [p["switched"] for p in self.ioi_prompts]
        )

    # ------------------------------------------------------------------
    # Convenience methods (mirrors IOIDataset interface)
    # ------------------------------------------------------------------

    def copy(self):
        return BiasedIOIDataset(
            N=self.N,
            switch_perc=self.switch_perc,
            bias_ratio=self.bias_ratio,
            tokenizer=self.tokenizer,
            prompts=copy.deepcopy(self.ioi_prompts),
            prepend_bos=self.prepend_bos,
            io_name=self.io_name,
            s_name=self.s_name,
        )

    def __len__(self):
        return self.N

    def __getitem__(self, key):
        sliced = self.ioi_prompts[key]
        return BiasedIOIDataset(
            N=len(sliced),
            switch_perc=self.switch_perc,
            bias_ratio=self.bias_ratio,
            tokenizer=self.tokenizer,
            prompts=sliced,
            prepend_bos=self.prepend_bos,
        )

    def __repr__(self):
        n_maj = self.majority_mask.sum().item()
        n_sw  = self.switched_mask.sum().item()
        return (
            f"BiasedIOIDataset(N={self.N}, "
            f"majority={n_maj} ({100*n_maj/self.N:.0f}% Mary=IO), "
            f"switched={n_sw} ({100*n_sw/self.N:.0f}%), "
            f"switch_perc={self.switch_perc})"
        )
