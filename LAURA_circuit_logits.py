"""
LAURA_circuit_logits.py

Fixed-circuit logit inspection using EasyTransformer's built-in ablation.
Runs independently — does NOT require any other LAURA_*.py output files.

Ablation strategy
-----------------
  Circuit heads are HARDCODED (26 heads from prior ablation results).
  All other 118 heads are mean-ablated using EasyTransformer's own machinery:

    EasyAblation   ← computes and caches mean activations
    abl.get_hook() ← returns the hook for each head using get_act_hook / cst_fn
                     (both from easy_transformer.experiments — no custom code)
    model.run_with_hooks() ← runs batched inference with the collected hooks

  head_circuit = "result" (hook_result, after W_O projection)
  use_attn_result = True is required.

Pipeline
--------
  1.  Build the biased IOI dataset  (1000 prompts)
  2.  Load GPT-2  (use_attn_result = True)
  3.  Circuit summary — visual grid of kept vs ablated heads
  4.  EasyAblation initialisation  (computes and caches mean activations)
  5.  Collect circuit hooks  (one per non-circuit head via abl.get_hook)
  6.  Logit inspection — Full GPT-2  (no ablation)
  7.  Logit inspection — Circuit-only  (118 heads mean-ablated)
  8.  Side-by-side comparison  (first 20 prompts)
  9.  Group-level summary

HOW TO RUN
----------
    python LAURA_circuit_logits.py > circuit_logits.out 2>circuit_progress.err
    tail -f circuit_progress.err
    less circuit_logits.out
"""

import os, sys

EASY_TRANSFORMER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Easy-Transformer"
)
sys.path.insert(0, EASY_TRANSFORMER_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

numb        = 1000
top_n       = 10
bias_ratio  = 0.7
switch_perc = 0.0
batch_size  = 32
seed        = 42

NAME_MAJORITY = "Mary"
NAME_MINORITY = "John"

# Hardcoded circuit: these 26 heads are KEPT active.
# Every other head is mean-ablated.
CIRCUIT_HEADS = [
    (9,  9), (10,  0), (9,  6), (10,  7), (11, 10),
    (8, 10), (7,   9), (8,  6), (7,   3), (5,   5),
    (5,  9), (6,   9), (5,  8), (0,   1), (0,  10),
    (3,  0), (4,  11), (2,  2), (11,  2), (10,  6),
    (10,10), (10,  2), (9,  7), (10,  1), (11,  9),
    (9,  0),
]

# EasyAblation caches means by running model(mean_dataset) as ONE batch.
# Cap this to avoid OOM — the metric itself still runs on all numb prompts
# via batched run_with_hooks calls below.
ABLATION_MEAN_N = 400

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from easy_transformer import EasyTransformer
from easy_transformer.experiments import (
    EasyAblation, AblationConfig, ExperimentMetric,
)
from easy_transformer.LAURA_biased_ioi_dataset import BiasedIOIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

W = 76
def rule(c="═"): print(c * W)
def section(t):  print(); rule("─"); print(f"  {t}"); rule("─")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_top_n(model, dataset, top_n, batch_size, device,
                  fwd_hooks=None, desc="forward passes"):
    """
    Run model.run_with_hooks on all prompts (batched).
    Returns a list of per-prompt dicts with logit-diff, IO rank, top-N tokens.
    fwd_hooks: list of (hook_name, hook_fn) pairs from abl.get_hook(), or None.
    """
    end_idxs  = dataset.word_idx["end"]
    io_ids    = torch.tensor(dataset.io_tokenIDs)
    s_ids     = torch.tensor(dataset.s_tokenIDs)
    io_names  = [p["IO"] for p in dataset.ioi_prompts]
    s_names   = [p["S"]  for p in dataset.ioi_prompts]
    N         = dataset.N
    tokenizer = dataset.tokenizer
    records   = []

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
            tnames = [tokenizer.decode(t.item()) for t in top_id]

            io_rank = next((k + 1 for k, t in enumerate(tnames)
                            if t.strip() == io_names[gi]), None)
            ld = lv[io_ids[gi]].item() - lv[s_ids[gi]].item()

            records.append(dict(
                idx=gi, io=io_names[gi], s=s_names[gi],
                prompt=dataset.sentences[gi],
                top_probs=tp, top_tokens=tnames,
                io_rank=io_rank, logit_diff=ld,
            ))

    return records


def print_logit_table(records, baseline_ld=None):
    io_correct = sum(1 for r in records if r["io_rank"] == 1)
    io_top5    = sum(1 for r in records if r["io_rank"] and r["io_rank"] <= 5)
    mean_ld    = sum(r["logit_diff"] for r in records) / len(records)
    N          = len(records)

    print(f"\n  N = {N}   IO correct (rank-1) = {io_correct}/{N} "
          f"({100*io_correct/N:.1f}%)   "
          f"IO in top-5 = {io_top5}/{N} ({100*io_top5/N:.1f}%)")
    print(f"  Mean logit-diff (IO − S) = {mean_ld:.4f}", end="")
    if baseline_ld is not None:
        pct = 100 * (mean_ld - baseline_ld) / abs(baseline_ld) if baseline_ld else 0
        print(f"   (baseline = {baseline_ld:.4f},  "
              f"change = {mean_ld - baseline_ld:+.4f} / {pct:+.1f}%)")
    else:
        print()

    print(f"\n  {'#':<6}  {'IO':^6}  {'S':^6}  "
          f"{'LD':>7}  {'Rank':^5}  "
          f"Top-{top_n} tokens  (renorm %)")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*7}  {'─'*5}  {'─'*55}")

    for r in records:
        rank_s  = str(r["io_rank"]) if r["io_rank"] else f">{top_n}"
        top_str = "  ".join(
            f"'{r['top_tokens'][k].strip()}' {r['top_probs'][k]*100:.1f}%"
            for k in range(min(top_n, len(r["top_tokens"]))))
        print(f"  {r['idx']+1:<6}  {r['io']:^6}  {r['s']:^6}  "
              f"{r['logit_diff']:>7.3f}  {rank_s:^5}  {top_str}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule()
print("  Biased IOI — Fixed-Circuit Logit Inspection")
print("  (EasyTransformer mean ablation via EasyAblation + get_act_hook)")
rule()
print(f"""
  numb={numb}  bias={bias_ratio:.0%} {NAME_MAJORITY}=IO  switch={switch_perc:.0%}  seed={seed}
  top_n={top_n}  batch_size={batch_size}
  circuit: {len(CIRCUIT_HEADS)} hardcoded heads kept
  ablated: {12*12 - len(CIRCUIT_HEADS)} heads mean-ablated  (hook_result, head_circuit=result)
  device={device}
""")
rule()

# ── 1. Dataset ────────────────────────────────────────────────────────────────
section("STEP 1 — BUILD DATASET")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Validate that both names tokenize to a single GPT-2 token.
# If a name is split into sub-tokens, io_tokenIDs will silently use only the
# first sub-token, and logit-diff / io_rank will be measured on the wrong token.
for role, name in [("NAME_MAJORITY", NAME_MAJORITY), ("NAME_MINORITY", NAME_MINORITY)]:
    ids = tokenizer.encode(" " + name)
    if len(ids) != 1:
        decoded = [tokenizer.decode([i]) for i in ids]
        raise ValueError(
            f"{role} = '{name}' tokenizes to {len(ids)} sub-tokens {decoded}. "
            f"Both names must be single GPT-2 tokens. "
            f"Safe choices include: Mary, John, Alice, Bob, Sarah, James, "
            f"Emma, Tom, Laura, Anna, Kate, Mark, Paul, Lisa, Ruth, Amy, Luke."
        )
print(f"\n  Name tokens OK:")
for name in [NAME_MAJORITY, NAME_MINORITY]:
    tid = tokenizer.encode(" " + name)[0]
    print(f"    '{name}' → token ID {tid}  ('{tokenizer.decode([tid])}')")

dataset = BiasedIOIDataset(
    N=numb, switch_perc=switch_perc, bias_ratio=bias_ratio,
    tokenizer=tokenizer, prepend_bos=True, seed=seed,
    io_name=NAME_MAJORITY, s_name=NAME_MINORITY,
)

n_maj = sum(1 for p in dataset.ioi_prompts if p["IO"] == NAME_MAJORITY)
n_min = numb - n_maj
print(f"\n  {dataset}")
print(f"  IO={NAME_MAJORITY}: {n_maj} prompts   IO={NAME_MINORITY}: {n_min} prompts")
print(f"\n  First 5 prompts:")
for i in range(min(5, numb)):
    p = dataset.ioi_prompts[i]
    print(f"    [{i+1}] IO={p['IO']:<5}  S={p['S']:<5}  {dataset.sentences[i][:75]}")

# ── 2. Load model ─────────────────────────────────────────────────────────────
section("STEP 2 — LOAD PRETRAINED GPT-2")
model = EasyTransformer.from_pretrained("gpt2")
model.cfg.use_attn_result = True   # required: registers hook_result hooks
model.to(device)
model.eval()
print(f"\n  GPT-2 small: {model.cfg.n_layers} layers × {model.cfg.n_heads} heads "
      f"({model.cfg.n_layers * model.cfg.n_heads} heads total)")
print(f"  use_attn_result = True  (hook_result hooks registered)")
print(f"  Device: {device}")

# ── 3. Circuit summary ────────────────────────────────────────────────────────
circuit_set = set(CIRCUIT_HEADS)
n_total   = model.cfg.n_layers * model.cfg.n_heads
n_ablated = n_total - len(CIRCUIT_HEADS)

section(f"STEP 3 — CIRCUIT  ({len(CIRCUIT_HEADS)} heads kept,  {n_ablated} mean-ablated)")

print(f"\n  Circuit heads by layer:")
by_layer = {}
for (l, h) in sorted(CIRCUIT_HEADS):
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

# ── 4. EasyAblation — compute and cache mean activations ─────────────────────
section("STEP 4 — EASY ABLATION INITIALISATION  (cache mean activations)")
print(f"""
  EasyAblation is used only for its mean-caching infrastructure here.
  We do NOT call run_ablation() — instead we call abl.get_hook(l, h) for
  each non-circuit head, which returns a hook built with EasyTransformer's
  own get_act_hook() / cst_fn() from experiments.py.

  mean_dataset = first {ABLATION_MEAN_N} prompts (pre-tokenized tensors)
    → get_all_mean() runs model(mean_dataset) as one forward pass
    → mean activations cached per hook point in abl.mean_cache
  head_circuit = "result"  (hook_result, shape [batch, seq, n_heads, d_model])
""", flush=True)

# Use pre-tokenized tensors so seq_len matches exactly what we use for inference.
# This avoids any mismatch from re-tokenizing strings with different padding.
mean_toks = dataset.toks[:ABLATION_MEAN_N].to(device)

# Dummy metric — we never call run_ablation(), so the metric function is not
# invoked. ExperimentMetric is required by EasyExperiment.__init__.
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
    head_circuit="result",
    cache_means=True,   # triggers get_all_mean() in __init__
    verbose=False,
)

print(f"  Initialising EasyAblation (running get_all_mean)...",
      file=sys.stderr, flush=True)
abl = EasyAblation(model, config, metric)
# abl.mean_cache now holds mean activations for all hook points
print(f"  Done.  {len(abl.mean_cache)} hook points cached.",
      file=sys.stderr, flush=True)

n_cached = len(abl.mean_cache)
print(f"\n  {n_cached} hook points in mean_cache.")
print(f"  Example keys (first 4): {list(abl.mean_cache.keys())[:4]}")

# ── 5. Collect circuit hooks (one per non-circuit head) ───────────────────────
section("STEP 5 — COLLECT CIRCUIT HOOKS  (via abl.get_hook)")
print(f"""
  For each head NOT in the circuit, call abl.get_hook(layer, head).
  This uses EasyTransformer's get_act_hook(cst_fn, mean, head, dim=2)
  internally — no custom hook code.

  The hook replaces  z[:, :, head, :]  with  mean[:batch, :seq, head, :]
  where mean is the cached dataset average for that hook point.
  cst_fn handles variable batch sizes via  cst[:z.shape[0], :z.shape[1]].
""")

circuit_hooks = []
for l in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        if (l, h) not in circuit_set:
            hook_name, hook_fn = abl.get_hook(l, h)
            circuit_hooks.append((hook_name, hook_fn))

print(f"  Collected {len(circuit_hooks)} hooks  "
      f"({n_ablated} non-circuit heads × 1 hook each).")

# Summarise by layer
print(f"\n  Hooks per layer:")
hooks_per_layer = {}
for hook_name, _ in circuit_hooks:
    # hook_name = "blocks.{l}.attn.hook_result"
    l = int(hook_name.split(".")[1])
    hooks_per_layer[l] = hooks_per_layer.get(l, 0) + 1
for l in range(model.cfg.n_layers):
    n_hooks = hooks_per_layer.get(l, 0)
    n_kept  = sum(1 for (ll, hh) in circuit_set if ll == l)
    print(f"    L{l:<2}  {n_hooks:2d} hooks (mean-ablated)  +  {n_kept:2d} kept")

# ── 6. Logit inspection — Full GPT-2 ─────────────────────────────────────────
section(f"STEP 6 — LOGIT INSPECTION: FULL GPT-2  (all {n_total} heads active)")
recs_full = collect_top_n(model, dataset, top_n, batch_size, device,
                           fwd_hooks=None, desc="full model")
baseline_ld = sum(r["logit_diff"] for r in recs_full) / len(recs_full)
print_logit_table(recs_full, baseline_ld=None)

# ── 7. Logit inspection — Circuit-only ───────────────────────────────────────
section(f"STEP 7 — LOGIT INSPECTION: CIRCUIT-ONLY  "
        f"({len(CIRCUIT_HEADS)} heads active,  {n_ablated} mean-ablated)")
print(f"\n  Running model.run_with_hooks on {numb} prompts "
      f"with {len(circuit_hooks)} ablation hooks...", flush=True)
recs_circuit = collect_top_n(model, dataset, top_n, batch_size, device,
                               fwd_hooks=circuit_hooks, desc="circuit-only")
print_logit_table(recs_circuit, baseline_ld=baseline_ld)

# ── 8. Side-by-side comparison ────────────────────────────────────────────────
section("STEP 8 — SIDE-BY-SIDE: FULL vs CIRCUIT-ONLY  (first 20 prompts)")
print(f"\n  {'#':<5}  {'IO':^5}  {'S':^5}  "
      f"{'LD_full':>9}  {'Rank_full':^10}  "
      f"{'LD_circ':>9}  {'Rank_circ':^10}  "
      f"{'Δ LD':>8}")
print(f"  {'─'*5}  {'─'*5}  {'─'*5}  "
      f"{'─'*9}  {'─'*10}  "
      f"{'─'*9}  {'─'*10}  "
      f"{'─'*8}")
for i in range(min(20, numb)):
    rf = recs_full[i]
    rc = recs_circuit[i]
    rf_rank = str(rf["io_rank"]) if rf["io_rank"] else f">{top_n}"
    rc_rank = str(rc["io_rank"]) if rc["io_rank"] else f">{top_n}"
    delta   = rc["logit_diff"] - rf["logit_diff"]
    print(f"  {i+1:<5}  {rf['io']:^5}  {rf['s']:^5}  "
          f"{rf['logit_diff']:>9.3f}  {rf_rank:^10}  "
          f"{rc['logit_diff']:>9.3f}  {rc_rank:^10}  "
          f"{delta:>+8.3f}")

# ── 9. Group-level summary ────────────────────────────────────────────────────
section("STEP 9 — GROUP-LEVEL SUMMARY")
for label, recs in [(f"Full GPT-2  (all {n_total} heads)",        recs_full),
                    (f"Circuit-only  ({len(CIRCUIT_HEADS)} heads)", recs_circuit)]:
    maj_ld = [r["logit_diff"] for r in recs if r["io"] == NAME_MAJORITY]
    min_ld = [r["logit_diff"] for r in recs if r["io"] == NAME_MINORITY]
    maj_r1 = sum(1 for r in recs if r["io"] == NAME_MAJORITY and r["io_rank"] == 1)
    min_r1 = sum(1 for r in recs if r["io"] == NAME_MINORITY and r["io_rank"] == 1)
    all_ld = (sum(maj_ld) + sum(min_ld)) / (len(maj_ld) + len(min_ld))

    print(f"\n  {label}")
    print(f"  {'─'*62}")
    print(f"  IO={NAME_MAJORITY} ({len(maj_ld)} prompts):"
          f"  mean LD = {sum(maj_ld)/len(maj_ld):+.4f}"
          f"  |  rank-1 = {maj_r1}/{len(maj_ld)} "
          f"({100*maj_r1/len(maj_ld):.1f}%)")
    print(f"  IO={NAME_MINORITY} ({len(min_ld)} prompts):"
          f"  mean LD = {sum(min_ld)/len(min_ld):+.4f}"
          f"  |  rank-1 = {min_r1}/{len(min_ld)} "
          f"({100*min_r1/len(min_ld):.1f}%)")
    print(f"  Overall:             "
          f"  mean LD = {all_ld:+.4f}"
          f"  |  rank-1 = {maj_r1+min_r1}/{numb} "
          f"({100*(maj_r1+min_r1)/numb:.1f}%)")

print()
rule()
print("  Done.")
rule()
