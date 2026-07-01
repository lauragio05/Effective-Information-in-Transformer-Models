"""
LAURA_biased_logits.py

Runs GPT-2 on the BiasedIOIDataset (70 % Mary=IO / 30 % John=IO) and records,
for every prompt, the top-N next-token predictions at the final "prediction
position" (the slot just before the model is expected to output the IO name,
i.e. after "... gave a <object> to").

No ablation is performed.  This is a clean forward-pass baseline.

HOW TO RUN
----------
    python LAURA_biased_logits.py > results.out 2>progress.err

    stdout  → results.out   (the formatted prompt list + summary)
    stderr  → progress.err  (tqdm progress bars — keeps the .out clean)

Or if you want everything in one file:
    python LAURA_biased_logits.py > results.out 2>&1

──────────────────────────────────────────────────────────────────────────────
CONFIG (edit here)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

# Path to the Easy-Transformer repo (one level up, sibling folder)
EASY_TRANSFORMER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Easy-Transformer"
)
sys.path.insert(0, EASY_TRANSFORMER_DIR)

# ── CONFIG ────────────────────────────────────────────────────────────────────

numb        = 100    # number of biased prompts to generate
top_n       = 10     # how many top predictions to show per prompt
switch_perc = 0.0    # fraction of prompts with name order swapped in first clause
bias_ratio  = 0.7    # fraction of prompts where Mary=IO / John=S
batch_size  = 32     # prompts per forward pass (lower if you run out of memory)

output_file = "biased_logits_results.pt"   # binary data file (saved alongside .out)

# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from easy_transformer import EasyTransformer
from easy_transformer.LAURA_biased_ioi_dataset import BiasedIOIDataset

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

print("Loading model...", flush=True)
model = EasyTransformer.from_pretrained("gpt2")
model.cfg.use_attn_result = False
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"  Model on {device}\n", flush=True)

# ── TOKENIZER ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── BUILD DATASET ─────────────────────────────────────────────────────────────

print(
    f"Generating {numb} prompts  "
    f"(bias={bias_ratio:.0%} Mary=IO  |  switch_perc={switch_perc:.0%})...",
    flush=True,
)

dataset = BiasedIOIDataset(
    N=numb,
    switch_perc=switch_perc,
    bias_ratio=bias_ratio,
    tokenizer=tokenizer,
    prepend_bos=True,
)
print(f"  {dataset}\n", flush=True)

# ── FORWARD PASSES ────────────────────────────────────────────────────────────
# word_idx["end"] is the token position at which the model should predict the
# IO name (i.e. the "to" token, last real token before the blank).

end_idxs = dataset.word_idx["end"]   # LongTensor [N]

all_top_logits  = []   # per-prompt: [top_n] float tensor
all_top_probs   = []   # per-prompt: [top_n] float tensor  (renormalised over top-n)
all_top_tok_ids = []   # per-prompt: [top_n] int tensor
all_top_tokens  = []   # per-prompt: list[str]

n_batches = (numb + batch_size - 1) // batch_size

for b in tqdm(range(n_batches), desc="forward passes", file=sys.stderr):
    start = b * batch_size
    end   = min(start + batch_size, numb)

    batch_toks = dataset.toks[start:end].to(device)   # [B, seq_len]
    batch_ends = end_idxs[start:end]                  # [B]

    with torch.no_grad():
        logits = model(batch_toks).cpu()              # [B, seq_len, vocab]

    for i in range(end - start):
        pos       = batch_ends[i].item()
        logit_vec = logits[i, pos, :]                 # [vocab]

        # softmax over the full vocabulary, then take top-n and renormalise
        # so the saved top_probs always sum to exactly 1 over the top-n window
        prob_vec              = F.softmax(logit_vec, dim=-1)
        top_vals, top_ids     = torch.topk(logit_vec, k=top_n)
        top_prob_vals         = prob_vec[top_ids]
        top_prob_vals         = top_prob_vals / top_prob_vals.sum()   # renormalise

        all_top_logits.append(top_vals)
        all_top_probs.append(top_prob_vals)
        all_top_tok_ids.append(top_ids)
        all_top_tokens.append(
            [tokenizer.decode(tid.item()) for tid in top_ids]
        )

# ── METADATA ──────────────────────────────────────────────────────────────────

io_names       = [p["IO"]       for p in dataset.ioi_prompts]
s_names        = [p["S"]        for p in dataset.ioi_prompts]
switched_flags = [p["switched"] for p in dataset.ioi_prompts]

io_ranks = []
for i in range(numb):
    io_id  = tokenizer.encode(" " + io_names[i])[0]
    toks_i = all_top_tok_ids[i].tolist()
    io_ranks.append(toks_i.index(io_id) + 1 if io_id in toks_i else None)

# ── SAVE BINARY RESULTS ───────────────────────────────────────────────────────

torch.save(
    {
        "prompts":        dataset.sentences,
        "io_names":       io_names,
        "s_names":        s_names,
        "switched":       switched_flags,
        "places":         [p["[PLACE]"]  for p in dataset.ioi_prompts],
        "objects":        [p["[OBJECT]"] for p in dataset.ioi_prompts],
        "top_logits":     torch.stack(all_top_logits,  dim=0),   # [N, top_n]
        "top_probs":      torch.stack(all_top_probs,   dim=0),   # [N, top_n]  renormalised → sums to 1
        "top_token_ids":  torch.stack(all_top_tok_ids, dim=0),   # [N, top_n]
        "top_tokens":     all_top_tokens,
        "io_ranks":       io_ranks,
        "end_idxs":       end_idxs,
        "config": {
            "numb": numb, "top_n": top_n,
            "switch_perc": switch_perc, "bias_ratio": bias_ratio,
            "model": "gpt2",
        },
    },
    output_file,
)

# ── FORMATTED OUTPUT (goes to stdout → .out file) ─────────────────────────────

W = 90   # total line width

def rule(char="═"):
    print(char * W)

def section(title):
    print()
    rule("─")
    print(f"  {title}")
    rule("─")

rule()
print(f"  GPT-2  |  BiasedIOIDataset  |  top-{top_n} predictions per prompt")
print(f"  numb={numb}  bias={bias_ratio:.0%} Mary=IO  switch_perc={switch_perc:.0%}")
rule()

# ── PER-PROMPT BLOCK ──────────────────────────────────────────────────────────

section(f"ALL PROMPTS  ({numb} total)")

for i in range(numb):
    sw_tag   = " [SWITCHED]" if switched_flags[i] else ""
    rank_str = f"rank {io_ranks[i]}" if io_ranks[i] is not None else f"outside top-{top_n}"

    print()
    print(f"[{i+1:03d}]  {dataset.sentences[i]}{sw_tag}")
    print(f"       IO={io_names[i]}  S={s_names[i]}  |  correct answer '{io_names[i]}' → {rank_str}")
    print(f"       {'Rank':<6}  {'Token':<18}  {'Logit':>8}  {'Prob':>8}")
    print(f"       {'────':<6}  {'─────':<18}  {'─────':>8}  {'────':>8}")

    for k in range(top_n):
        tok      = repr(all_top_tokens[i][k])          # shows spaces clearly
        logit    = all_top_logits[i][k].item()
        prob_pct = all_top_probs[i][k].item() * 100

        # mark the row if it's the correct IO token
        is_io  = all_top_tokens[i][k].strip() == io_names[i]
        marker = " ◄" if is_io else ""

        print(f"       {k+1:<6}  {tok:<18}  {logit:>8.3f}  {prob_pct:>7.2f}%{marker}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

def pct(x, total):
    return f"{100*x/total:.1f}%" if total else "n/a"

n_found    = sum(r is not None for r in io_ranks)
n_rank1    = sum(r == 1        for r in io_ranks if r is not None)
n_majority = sum(io == "Mary"  for io in io_names)
n_switched = sum(switched_flags)

section("SUMMARY")
print(f"  Total prompts         : {numb}")
print(f"  Mary=IO (majority)    : {n_majority}  ({pct(n_majority, numb)})")
print(f"  John=IO (minority)    : {numb-n_majority}  ({pct(numb-n_majority, numb)})")
print(f"  Switched order        : {n_switched}  ({pct(n_switched, numb)})")
print(f"  IO in top-{top_n}          : {n_found}  ({pct(n_found, numb)})")
print(f"  IO ranked #1          : {n_rank1}  ({pct(n_rank1, numb)})")

section("BREAKDOWN BY SUBGROUP")
groups = [
    ("All prompts",          lambda i: True),
    ("Mary=IO  (majority)",  lambda i: io_names[i] == "Mary"),
    ("John=IO  (minority)",  lambda i: io_names[i] == "John"),
    ("Normal order",         lambda i: not switched_flags[i]),
    ("Switched order",       lambda i: switched_flags[i]),
]
for label, fn in groups:
    idx = [i for i in range(numb) if fn(i)]
    if not idx:
        continue
    found = sum(1 for i in idx if io_ranks[i] is not None)
    r1    = sum(1 for i in idx if io_ranks[i] == 1)
    io_p  = [all_top_probs[i][io_ranks[i]-1].item()*100
             for i in idx if io_ranks[i] is not None]
    mean_p = sum(io_p)/len(io_p) if io_p else float("nan")
    print(f"  {label:<25}  n={len(idx):>4}  "
          f"in top-{top_n}: {pct(found,len(idx)):>6}  "
          f"rank-1: {pct(r1,len(idx)):>6}  "
          f"mean IO prob: {mean_p:>5.2f}%")

print()
rule()
print(f"  Binary results saved → {output_file}")
rule()
