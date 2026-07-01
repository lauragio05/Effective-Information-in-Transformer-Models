"""
LAURA_full_experiment.py

Unified experiment runner.  Generates one biased IOI dataset and runs the
full pipeline (logits → macro states → EI / Eff / ΔEff) for three model
conditions:

    1. Full GPT-2          — all heads active, pretrained weights
    2. Circuit-only GPT-2  — 26 hardcoded circuit heads kept active,
                             all other 118 heads mean-ablated
    3. Untrained GPT-2     — same GPT-2 architecture, weights re-initialised
                             randomly (never trained on any data)

All three conditions share the same dataset.

ABLATION APPROACH  (mirrors LAURA_circuit_logits.py exactly)
-----------------
  Circuit heads are HARDCODED in CIRCUIT_HEADS below.
  Mean ablation uses EasyTransformer's built-in EasyAblation infrastructure:
    • use_attn_result = True  (registers hook_result hooks per head)
    • head_circuit = "result" (ablation hooks act on hook_result)
    • EasyAblation.get_hook(layer, head) returns the hook for each head
      via get_act_hook / cst_fn from easy_transformer.experiments
  Logit extraction is identical to LAURA_circuit_logits.py.
  The only addition is the EI / macro-state calculation that follows.

MODEL NOTES
-----------
  Pretrained GPT-2 : trained by OpenAI (2019) on ~40 GB of internet text.
    `EasyTransformer.from_pretrained("gpt2")` downloads saved weights from
    HuggingFace — no training happens here.
  Untrained GPT-2  : identical architecture, all weights re-initialised
    from scratch (Normal(0, 0.02) / zeros) — has never seen any data.

HOW TO RUN
----------
    python LAURA_full_experiment.py > full_experiment.out 2>progress.err
    stdout → full_experiment.out   (all results, tables, per-prompt detail)
    stderr → progress.err          (tqdm progress bars — keeps .out clean)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATHS  ← set these for your cluster
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys

# ── CHANGE THIS to the absolute path of your Easy-Transformer folder ──────────
EASY_TRANSFORMER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Easy-Transformer"
)   # e.g.  "/home/lpaxton/Easy-Transformer"

sys.path.insert(0, EASY_TRANSFORMER_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG  ← edit freely
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

numb        = 1000   # total number of prompts (shared across all conditions)
top_n       = 10     # top-K tokens kept at the micro level
bias_ratio  = 0.7    # fraction of prompts where NAME_MAJORITY is IO
switch_perc = 0.0    # fraction of prompts with swapped first-clause order
batch_size  = 32     # prompts per forward pass (lower if OOM)
seed        = 42

# Names used in the biased dataset (any two names from ioi_dataset.NAMES)
NAME_MAJORITY = "Mary"   # IO in `bias_ratio` fraction of prompts
NAME_MINORITY = "John"   # IO in the remaining fraction

# Hardcoded circuit: these 26 heads are KEPT active.
# Every other head is mean-ablated (118 heads total ablated).
# Taken directly from LAURA_circuit_logits.py.
CIRCUIT_HEADS = [
    (9,  9), (10,  0), (9,  6), (10,  7), (11, 10),
    (8, 10), (7,   9), (8,  6), (7,   3), (5,   5),
    (5,  9), (6,   9), (5,  8), (0,   1), (0,  10),
    (3,  0), (4,  11), (2,  2), (11,  2), (10,  6),
    (10,10), (10,  2), (9,  7), (10,  1), (11,  9),
    (9,  0),
]

# EasyAblation caches means by running model(mean_dataset) as ONE forward pass.
# Cap this to avoid OOM — inference still runs on all numb prompts via batched
# run_with_hooks calls.
ABLATION_MEAN_N = 400

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math, copy, time, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from easy_transformer import EasyTransformer
from easy_transformer.experiments import (
    EasyAblation, AblationConfig, ExperimentMetric,
)
from easy_transformer.LAURA_biased_ioi_dataset import BiasedIOIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def entropy(dist):
    return -sum(p * math.log2(p) for p in dist if p > 0)

def avg_dist(dists):
    k, n = len(dists[0]), len(dists)
    return [sum(d[j] for d in dists) / n for j in range(k)]

def compute_ei(dist_A, dist_B):
    ue   = [(a + b) / 2 for a, b in zip(dist_A, dist_B)]
    H_UE = entropy(ue)
    H_A  = entropy(dist_A)
    H_B  = entropy(dist_B)
    EI   = max(H_UE - (H_A + H_B) / 2, 0.0)
    Eff  = EI / math.log2(len(dist_A)) if len(dist_A) > 1 else 0.0
    return EI, Eff, H_UE, H_A, H_B

def macro_dist_from_row(probs, tokens):
    pm = pn = 0.0
    for k, tok in enumerate(tokens):
        t = tok.strip()
        if t == NAME_MAJORITY: pm += probs[k]
        elif t == NAME_MINORITY: pn += probs[k]
    return [pm, pn, max(0.0, 1.0 - pm - pn)]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABLATION HELPERS  (mirrors LAURA_circuit_logits.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_easy_ablation(model, dataset, mean_n, device):
    """
    Initialise EasyAblation for mean-ablation via hook_result.

    Requires model.cfg.use_attn_result = True.
    Caches mean activations from the first `mean_n` prompts.
    Returns an EasyAblation instance whose .get_hook(layer, head)
    yields the (hook_name, hook_fn) pair for that head.
    """
    mean_toks = dataset.toks[:mean_n].to(device)

    # Dummy metric — run_ablation() is never called, only get_hook().
    def _dummy_metric(model, data):
        return torch.tensor(0.0)

    metric = ExperimentMetric(
        metric=_dummy_metric,
        dataset=mean_toks,
        relative_metric=False,
        scalar_metric=True,
    )
    config = AblationConfig(
        abl_type="mean",
        mean_dataset=mean_toks,
        target_module="attn_head",
        head_circuit="result",   # hooks on hook_result
        cache_means=True,        # triggers get_all_mean() in __init__
        verbose=False,
    )
    return EasyAblation(model, config, metric)


def build_circuit_hooks(model, abl, circuit_set):
    """
    For every head NOT in circuit_set, call abl.get_hook(l, h) to get a
    hook that replaces that head's hook_result with its cached mean.
    Returns the list of (hook_name, hook_fn) pairs.
    """
    hooks = []
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            if (l, h) not in circuit_set:
                hook_name, hook_fn = abl.get_hook(l, h)
                hooks.append((hook_name, hook_fn))
    return hooks

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE PIPELINE  (logit extraction → macro states → EI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(model, dataset, top_n, batch_size, device,
                 fwd_hooks=None, desc="forward passes"):
    """
    Run batched forward passes (with optional ablation hooks), collect
    top-N logits per prompt, compute macro distributions, then compute EI.

    fwd_hooks : list of (hook_name, hook_fn) from build_circuit_hooks(),
                or None for the full / untrained model conditions.
    """
    end_idxs  = dataset.word_idx["end"]
    io_names  = [p["IO"] for p in dataset.ioi_prompts]
    s_names   = [p["S"]  for p in dataset.ioi_prompts]
    io_ids    = torch.tensor(dataset.io_tokenIDs)
    s_ids     = torch.tensor(dataset.s_tokenIDs)
    N         = dataset.N
    tokenizer = dataset.tokenizer

    top_probs_all, top_tokens_all, logit_diffs = [], [], []

    for b in tqdm(range((N + batch_size - 1) // batch_size),
                  desc=f"  {desc}", file=sys.stderr, leave=False):
        start = b * batch_size
        end_b = min(start + batch_size, N)
        batch = dataset.toks[start:end_b].to(device)

        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(batch, fwd_hooks=fwd_hooks).cpu()
            else:
                logits = model(batch).cpu()

        for i in range(end_b - start):
            gi  = start + i
            pos = end_idxs[gi].item()
            lv  = logits[i, pos, :]
            pv  = F.softmax(lv, dim=-1)

            top_v, top_id = torch.topk(lv, k=top_n)
            tp = pv[top_id]
            tp = (tp / tp.sum()).tolist()

            top_probs_all.append(tp)
            top_tokens_all.append([tokenizer.decode(t.item()) for t in top_id])
            logit_diffs.append(lv[io_ids[gi]].item() - lv[s_ids[gi]].item())

    micro_maj, micro_min = [], []
    macro_maj, macro_min = [], []
    per_prompt = []

    for i in range(N):
        probs   = top_probs_all[i]
        toks    = top_tokens_all[i]
        macro   = macro_dist_from_row(probs, toks)
        io_rank = next((k + 1 for k, t in enumerate(toks)
                        if t.strip() == io_names[i]), None)
        per_prompt.append({
            "text":       dataset.sentences[i],
            "io":         io_names[i],
            "s":          s_names[i],
            "probs":      probs,
            "tokens":     toks,
            "macro":      macro,
            "io_rank":    io_rank,
            "logit_diff": logit_diffs[i],
        })
        if io_names[i] == NAME_MAJORITY:
            micro_maj.append(probs); macro_maj.append(macro)
        else:
            micro_min.append(probs); macro_min.append(macro)

    tpm_maj_micro = avg_dist(micro_maj)
    tpm_min_micro = avg_dist(micro_min)
    tpm_maj_macro = avg_dist(macro_maj)
    tpm_min_macro = avg_dist(macro_min)

    EI_micro, Eff_micro, H_UE_mi, H_maj_m, H_min_m = compute_ei(tpm_maj_micro, tpm_min_micro)
    EI_macro, Eff_macro, H_UE_ma, H_maj_M, H_min_M = compute_ei(tpm_maj_macro, tpm_min_macro)

    mean_ld = sum(logit_diffs) / len(logit_diffs)

    return dict(
        tpm_majority_micro=tpm_maj_micro, tpm_minority_micro=tpm_min_micro,
        tpm_majority_macro=tpm_maj_macro, tpm_minority_macro=tpm_min_macro,
        H_UE_micro=H_UE_mi,  H_UE_macro=H_UE_ma,
        H_maj_micro=H_maj_m, H_min_micro=H_min_m,
        H_maj_macro=H_maj_M, H_min_macro=H_min_M,
        ei_micro=EI_micro,   eff_micro=Eff_micro,
        ei_macro=EI_macro,   eff_macro=Eff_macro,
        delta_eff=Eff_macro - Eff_micro,
        mean_logit_diff=mean_ld,
        n_majority=len(micro_maj), n_minority=len(micro_min),
        per_prompt=per_prompt,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

W = 72
def rule(c="═"): print(c * W)
def section(t):  print(); rule("─"); print(f"  {t}"); rule("─")

def print_circuit_summary(circuit_set, model):
    n_total   = model.cfg.n_layers * model.cfg.n_heads
    n_ablated = n_total - len(circuit_set)
    section(f"CIRCUIT  ({len(circuit_set)} heads kept,  {n_ablated} mean-ablated)")

    print(f"\n  Circuit heads by layer:")
    by_layer = {}
    for (l, h) in sorted(circuit_set):
        by_layer.setdefault(l, []).append(h)
    for l in sorted(by_layer):
        heads_str = "  ".join(f"H{h}" for h in sorted(by_layer[l]))
        print(f"    L{l:<2}  ({len(by_layer[l])} heads):  {heads_str}")

    empty_layers = sorted(set(range(model.cfg.n_layers)) - set(by_layer))
    if empty_layers:
        print(f"\n  Layers with NO circuit heads (fully ablated): "
              f"{[f'L{l}' for l in empty_layers]}")

    print(f"\n  Grid  (● = kept,  · = mean-ablated):\n")
    print("       " + "".join(f"  H{h:<2}" for h in range(model.cfg.n_heads)))
    for l in range(model.cfg.n_layers):
        row = f"  L{l:<2}  " + "".join(
            "  ●  " if (l, h) in circuit_set else "  ·  "
            for h in range(model.cfg.n_heads))
        print(row)

def print_condition_results(label, r, top_n):
    section(f"CONDITION: {label}")
    n_maj, n_min = r["n_majority"], r["n_minority"]

    # logit-diff summary
    io_correct = sum(1 for p in r["per_prompt"] if p["io_rank"] == 1)
    io_top5    = sum(1 for p in r["per_prompt"] if p["io_rank"] and p["io_rank"] <= 5)
    N          = len(r["per_prompt"])
    print(f"\n  Mean logit-diff (IO − S) = {r['mean_logit_diff']:.4f}")
    print(f"  IO rank-1 = {io_correct}/{N} ({100*io_correct/N:.1f}%)   "
          f"IO in top-5 = {io_top5}/{N} ({100*io_top5/N:.1f}%)")

    # macro TPM table
    print(f"\n  Macro TPM  →  {{{NAME_MAJORITY}, {NAME_MINORITY}, Other}}")
    print(f"\n  {'Group':<22}  {f'P({NAME_MAJORITY})':>10}  "
          f"{f'P({NAME_MINORITY})':>10}  {'P(Other)':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*10}")
    for name, tpm in [(f"{NAME_MAJORITY}=IO  (n={n_maj})", r["tpm_majority_macro"]),
                      (f"{NAME_MINORITY}=IO  (n={n_min})", r["tpm_minority_macro"])]:
        print(f"  {name:<22}  {tpm[0]*100:>9.3f}%  {tpm[1]*100:>9.3f}%  {tpm[2]*100:>9.3f}%")

    # entropy table
    print(f"\n  Entropy (bits){'':20}  {'Micro':>8}  {'Macro':>8}")
    print(f"  {'─'*34}  {'─'*8}  {'─'*8}")
    for lbl, vm, vM in [
        (f"H(effect | {NAME_MAJORITY}=IO)", r["H_maj_micro"], r["H_maj_macro"]),
        (f"H(effect | {NAME_MINORITY}=IO)", r["H_min_micro"], r["H_min_macro"]),
        ("H(UE) — marginal effect",         r["H_UE_micro"],  r["H_UE_macro"]),
    ]:
        print(f"  {lbl:<34}  {vm:>8.4f}  {vM:>8.4f}")

    # EI table
    print(f"\n  {'Metric':<38}  {'Micro':>10}  {'Macro':>10}")
    print(f"  {'─'*38}  {'─'*10}  {'─'*10}")
    for lbl, vm, vM in [
        ("EI  (bits)",              r["ei_micro"],    r["ei_macro"]),
        ("max EI = log2(n_states)", math.log2(top_n), math.log2(3)),
        ("Eff = EI / log2(n)",      r["eff_micro"],   r["eff_macro"]),
    ]:
        print(f"  {lbl:<38}  {vm:>10.4f}  {vM:>10.4f}")
    print(f"  {'─'*38}  {'─'*10}  {'─'*10}")
    print(f"  {'ΔEff  (macro − micro)':<38}  {r['delta_eff']:>+10.4f}")
    verdict = ("CAUSAL EMERGENCE" if r["delta_eff"] > 0 else
               "CAUSAL REDUCTION" if r["delta_eff"] < 0 else "NO CHANGE")
    print(f"\n  → {verdict}")

    # per-prompt detail (first 20)
    print(f"\n  Per-prompt predictions  (first 20 shown)\n")
    print(f"  {'#':<6}  {'IO':^6}  {'S':^6}  {'LD':>7}  {'Rank':^5}  "
          f"Top-{min(5,top_n)} tokens  (renorm %)")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*50}")
    for j, p in enumerate(r["per_prompt"][:20]):
        rank_s = str(p["io_rank"]) if p["io_rank"] else f">{top_n}"
        top5   = "  ".join(f"'{p['tokens'][k].strip()}' {p['probs'][k]*100:.1f}%"
                           for k in range(min(5, top_n)))
        print(f"  {j+1:<6}  {p['io']:^6}  {p['s']:^6}  "
              f"{p['logit_diff']:>7.3f}  {rank_s:^5}  {top5}")

def print_comparison_table(labels, results):
    section("COMPARISON ACROSS CONDITIONS")
    print(f"\n  {'Condition':<40}  {'mean LD':>8}  {'EI_mi':>7}  {'Eff_mi':>7}  "
          f"{'EI_ma':>7}  {'Eff_ma':>7}  {'ΔEff':>8}  Verdict")
    print(f"  {'─'*40}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*16}")
    for label, r in zip(labels, results):
        v = ("EMERGENCE" if r["delta_eff"] > 0 else
             "REDUCTION" if r["delta_eff"] < 0 else "NEUTRAL")
        print(f"  {label:<40}  {r['mean_logit_diff']:>8.4f}  "
              f"{r['ei_micro']:>7.4f}  {r['eff_micro']:>7.4f}  "
              f"{r['ei_macro']:>7.4f}  {r['eff_macro']:>7.4f}  "
              f"{r['delta_eff']:>+8.4f}  {v}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule()
print("  Biased IOI Experiment  —  Effective Information Analysis")
rule()
print(f"""
  Dataset      numb={numb}  bias={bias_ratio:.0%} {NAME_MAJORITY}=IO  switch={switch_perc:.0%}  seed={seed}
  Model        GPT-2 small  (12 layers × 12 heads = 144 heads total)
  Circuit      {len(CIRCUIT_HEADS)} hardcoded heads kept  |  {144 - len(CIRCUIT_HEADS)} mean-ablated
  top_n        {top_n}  micro-level tokens
  Ablation     EasyAblation / hook_result  (use_attn_result=True)

  GPT-2 pretrained weights : downloaded from HuggingFace (OpenAI, 2019,
    trained on ~40 GB WebText).  No training happens in this script.
  Untrained model : same architecture, all weights re-initialised randomly
    (Normal(0, 0.02) / zeros) — has never seen any data.
""")
rule()

# ── Dataset (shared across all conditions) ────────────────────────────────────
print("Building dataset...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Validate single-token names (same check as LAURA_circuit_logits.py)
for role, name in [("NAME_MAJORITY", NAME_MAJORITY), ("NAME_MINORITY", NAME_MINORITY)]:
    ids = tokenizer.encode(" " + name)
    if len(ids) != 1:
        decoded = [tokenizer.decode([i]) for i in ids]
        raise ValueError(
            f"{role} = '{name}' tokenizes to {len(ids)} sub-tokens {decoded}. "
            f"Both names must be single GPT-2 tokens. "
            f"Safe choices: Mary, John, Alice, Bob, Sarah, James, Emma, Tom, "
            f"Laura, Anna, Kate, Mark, Paul, Lisa, Ruth, Amy, Luke."
        )

dataset = BiasedIOIDataset(
    N=numb, switch_perc=switch_perc, bias_ratio=bias_ratio,
    tokenizer=tokenizer, prepend_bos=True, seed=seed,
)
n_maj = sum(1 for p in dataset.ioi_prompts if p["IO"] == NAME_MAJORITY)
n_min = numb - n_maj
print(f"  {dataset}")
print(f"  IO={NAME_MAJORITY}: {n_maj} prompts   IO={NAME_MINORITY}: {n_min} prompts\n",
      flush=True)

# ── Load pretrained model ─────────────────────────────────────────────────────
print("Loading pretrained GPT-2...", flush=True)
model_full = EasyTransformer.from_pretrained("gpt2")
model_full.cfg.use_attn_result = True   # required for hook_result / EasyAblation
model_full.to(device)
model_full.eval()
print(f"  on {device}\n", flush=True)

# ── Circuit summary ───────────────────────────────────────────────────────────
circuit_set = set(CIRCUIT_HEADS)
print_circuit_summary(circuit_set, model_full)

# ── EasyAblation — cache mean activations, build circuit hooks ────────────────
print("\nInitialising EasyAblation (caching mean activations)...",
      file=sys.stderr, flush=True)
abl = init_easy_ablation(model_full, dataset, ABLATION_MEAN_N, device)
print(f"  {len(abl.mean_cache)} hook points cached.",
      file=sys.stderr, flush=True)

circuit_hooks = build_circuit_hooks(model_full, abl, circuit_set)
print(f"  {len(circuit_hooks)} ablation hooks built "
      f"({144 - len(CIRCUIT_HEADS)} non-circuit heads).",
      file=sys.stderr, flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONDITION 1 — Full GPT-2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Running condition 1 — Full GPT-2...", file=sys.stderr, flush=True)
res_full = run_pipeline(model_full, dataset, top_n, batch_size, device,
                        fwd_hooks=None, desc="condition 1: full GPT-2")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONDITION 2 — Circuit-only  (118 heads mean-ablated via EasyAblation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Running condition 2 — Circuit-only...", file=sys.stderr, flush=True)
res_circuit = run_pipeline(model_full, dataset, top_n, batch_size, device,
                           fwd_hooks=circuit_hooks,
                           desc="condition 2: circuit-only")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONDITION 3 — Untrained GPT-2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Building untrained model...", file=sys.stderr, flush=True)

model_untrained = copy.deepcopy(model_full)

# Seed from current time (millisecond precision) → different random weights
# every run.  Printed to stdout so results remain reproducible if needed.
untrained_seed = int(time.time() * 1000) % (2**31)
print(f"  Untrained model seed: {untrained_seed}  "
      f"(time-based — different every run)", flush=True)
torch.manual_seed(untrained_seed)

# Re-initialise ALL parameters.
# EasyTransformer stores weights as raw nn.Parameters, not inside nn.Linear /
# nn.Embedding, so model.apply() with isinstance checks never matches.
# We iterate named_parameters() directly and handle every name explicitly.
#
# Handling rules:
#   LayerNorm scale  (last == "w", 1-D) → 1.0   (neutral, no learned scaling)
#   LayerNorm bias   (last == "b")       → 0.0   (neutral, no learned shift)
#   Attention biases (last starts "b_")  → N(0, 0.02)  ← fully random
#   All weight matrices                  → N(0, 0.02)  ← fully random
#
# Buffers (causal mask, IGNORE sentinel) are registered via register_buffer,
# NOT via nn.Parameter, so they do NOT appear in named_parameters() and are
# intentionally left untouched — they are structural constants, not weights.

handled = {"layernorm_scale": [], "layernorm_bias": [], "randomised": []}

with torch.no_grad():
    for name, param in model_untrained.named_parameters():
        last = name.split(".")[-1]

        if last == "w" and param.dim() == 1:
            # LayerNorm scale: keep at 1.0 so normalisation is neutral
            param.fill_(1.0)
            handled["layernorm_scale"].append(name)

        elif last == "b" and param.dim() == 1:
            # LayerNorm bias: keep at 0.0 so normalisation is neutral
            param.zero_()
            handled["layernorm_bias"].append(name)

        else:
            # Everything else — weight matrices AND attention/MLP biases —
            # fully randomised with N(0, 0.02)
            param.normal_(0.0, 0.02)
            handled["randomised"].append(name)

# ── Diagnostic: print every parameter name so coverage can be verified ────────
section("UNTRAINED MODEL — PARAMETER INITIALISATION AUDIT")
print(f"\n  Seed: {untrained_seed}")
print(f"\n  LayerNorm scale  → 1.0  ({len(handled['layernorm_scale'])} params)")
for n in handled["layernorm_scale"]:
    print(f"    {n}")
print(f"\n  LayerNorm bias   → 0.0  ({len(handled['layernorm_bias'])} params)")
for n in handled["layernorm_bias"]:
    print(f"    {n}")
print(f"\n  Randomised N(0,0.02)  ({len(handled['randomised'])} params)")
for n in handled["randomised"]:
    p = dict(model_untrained.named_parameters())[n]
    print(f"    {n:<55}  shape={list(p.shape)}")

total = sum(len(v) for v in handled.values())
n_params_total = sum(p.numel() for p in model_untrained.parameters())
print(f"\n  Total parameter tensors accounted for: {total}")
print(f"  Total scalar parameters: {n_params_total:,}")

model_untrained.to(device)
model_untrained.eval()

print("Running condition 3 — Untrained GPT-2...", file=sys.stderr, flush=True)
res_untrained = run_pipeline(model_untrained, dataset, top_n, batch_size, device,
                             fwd_hooks=None, desc="condition 3: untrained")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRINT ALL RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

labels  = [
    "Full GPT-2",
    f"Circuit-only GPT-2  ({len(CIRCUIT_HEADS)} heads)",
    "Untrained GPT-2",
]
results = [res_full, res_circuit, res_untrained]

for label, r in zip(labels, results):
    print_condition_results(label, r, top_n)
    print()

print_comparison_table(labels, results)

print()
rule()
print("  Done.")
rule()
