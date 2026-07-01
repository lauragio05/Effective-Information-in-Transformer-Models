"""
LAURA_ablation_study.py

Standalone ablation + logit inspection script using EasyAblation.
Runs independently — does NOT require any other LAURA_*.py output files.

Ablation strategy (identical to LAURA_run_ablation.py)
-------------------------------------------------------
  EasyAblation with:
    abl_type    = "mean"
    target_module = "attn_head"
    head_circuit  = "result"   ← requires use_attn_result = True
    cache_means   = True
    relative_metric = True     ← scores are (ablated - baseline) / |baseline|

  semantic_indices are derived from the biased IOI dataset so the mean
  ablation respects the correct token positions (IO, S, S2, end).

Pipeline
--------
  1. Build the biased IOI dataset
  2. Load GPT-2 with use_attn_result = True
  3. Build semantic_indices from dataset.word_idx
  4. Run EasyAblation → score matrix [n_layers × n_heads]
  5. Select circuit heads by threshold
  6. Compute hook_result means (for building circuit hooks)
  7. Run forward passes (full model + circuit-only) → per-prompt top-N predictions

No EI / macro-state computation — this file is for verifying the ablation
and checking that the model's predictions look sensible.

HOW TO RUN
----------
    python LAURA_ablation_study.py > ablation_study.out 2>ablation_progress.err
    tail -f ablation_progress.err      ← watch progress
    less ablation_study.out            ← inspect results
"""

import os, sys

EASY_TRANSFORMER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Easy-Transformer"
)
sys.path.insert(0, EASY_TRANSFORMER_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

numb               = 1000   # total number of prompts
top_n              = 10     # top-K tokens shown per prompt
bias_ratio         = 0.7    # fraction of prompts where NAME_MAJORITY is IO
switch_perc        = 0.0    # fraction with swapped first-clause order
batch_size         = 32
seed               = 42

NAME_MAJORITY = "Mary"
NAME_MINORITY = "John"

ABLATION_THRESHOLD = -0.05  # heads with score < this are kept as "circuit"

# EasyAblation's get_all_mean() runs model(mean_dataset) as ONE batch.
# If mean_dataset is too large it OOMs.  We cap it here; our own
# compute_head_means() in Step 5 uses proper batching over the full dataset.
ABLATION_MEAN_N    = 400    # max prompts passed to EasyAblation for mean computation

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from easy_transformer import EasyTransformer
from easy_transformer.experiments import EasyAblation, AblationConfig, ExperimentMetric
from easy_transformer.LAURA_biased_ioi_dataset import BiasedIOIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

W = 76
def rule(c="═"): print(c * W)
def section(t):  print(); rule("─"); print(f"  {t}"); rule("─")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABLATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_head_means(model, toks, batch_size, device):
    """
    Average hook_result (head output AFTER output projection) over all
    prompts and all sequence positions.
    Returns [n_layers, n_heads, d_model].

    hook_result requires use_attn_result = True (already set in this script).
    Used to build the circuit hooks that mean-ablate non-circuit heads.
    """
    n_l  = model.cfg.n_layers
    n_h  = model.cfg.n_heads
    d_m  = model.cfg.d_model
    sums = torch.zeros(n_l, n_h, d_m)
    cnt  = 0
    for start in tqdm(range(0, toks.shape[0], batch_size),
                      desc="  computing head means", file=sys.stderr, leave=False):
        batch = toks[start:start+batch_size].to(device)
        cap   = {}
        hooks = [
            (f"blocks.{l}.attn.hook_result",
             (lambda layer: lambda r, hook=None: cap.__setitem__(layer, r.detach().cpu()))(l))
            for l in range(n_l)
        ]
        with torch.no_grad():
            model.run_with_hooks(batch, fwd_hooks=hooks)
        for l in range(n_l):
            # cap[l] shape: [batch, seq_len, n_heads, d_model]
            sums[l] += cap[l].sum(dim=[0, 1])
            cnt      += batch.shape[0] * cap[l].shape[1]
    return sums / cnt


def select_circuit_heads(scores, threshold):
    """Return list of (layer, head) tuples with score < threshold."""
    return [(l, h)
            for l in range(scores.shape[0])
            for h in range(scores.shape[1])
            if scores[l, h].item() < threshold]


def build_ablation_hooks(model, relevant_set, head_means, device):
    """
    One hook per layer on hook_result.
    Every head NOT in relevant_set has its result replaced by its dataset mean.
    """
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    hooks = []
    for l in range(n_l):
        to_ablate = [h for h in range(n_h) if (l, h) not in relevant_set]
        if not to_ablate:
            continue
        lm = head_means[l].to(device)   # [n_heads, d_model]

        def make_layer_hook(heads, means):
            def fn(result, hook=None):
                result = result.clone()
                for h in heads:
                    result[:, :, h, :] = means[h].unsqueeze(0).unsqueeze(0).expand(
                        result.shape[0], result.shape[1], -1)
                return result
            return fn

        hooks.append((f"blocks.{l}.attn.hook_result",
                      make_layer_hook(to_ablate, lm)))
    return hooks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGIT / TOKEN PREDICTION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_top_n(model, dataset, top_n, batch_size, device,
                  ablation_hooks=None, desc="forward passes"):
    """
    Run the model (with optional ablation hooks on hook_result) and collect
    per-prompt:
      top_probs  : renormalised top-N probabilities
      top_tokens : decoded token strings
      io_rank    : rank of the IO token (1 = top prediction), None if not in top-N
      logit_diff : logit(IO) − logit(S) at the prediction position
    """
    end_idxs = dataset.word_idx["end"]
    io_ids   = torch.tensor(dataset.io_tokenIDs)
    s_ids    = torch.tensor(dataset.s_tokenIDs)
    io_names = [p["IO"] for p in dataset.ioi_prompts]
    s_names  = [p["S"]  for p in dataset.ioi_prompts]
    N        = dataset.N
    tokenizer = dataset.tokenizer
    records  = []

    for b in tqdm(range((N + batch_size - 1) // batch_size),
                  desc=f"  {desc}", file=sys.stderr, leave=False):
        start = b * batch_size
        end_b = min(start + batch_size, N)
        batch = dataset.toks[start:end_b].to(device)
        with torch.no_grad():
            logits = (model.run_with_hooks(batch, fwd_hooks=ablation_hooks).cpu()
                      if ablation_hooks else model(batch).cpu())

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_score_matrix(scores, n_l, n_h):
    print(f"\n  Rows = layers (L0–L{n_l-1})   Cols = heads (H0–H{n_h-1})")
    print(f"  Values = relative change in logit-diff when that head is mean-ablated.")
    print(f"  0.00 = no effect   −1.00 = logit-diff halved   +0.xx = head hurts IOI\n")
    header = "       " + "".join(f"  H{h:<2}" for h in range(n_h))
    print(header)
    for l in range(n_l):
        row = f"  L{l:<2}  " + "".join(
            f"{scores[l,h].item():+.2f} " for h in range(n_h))
        print(row)


def print_head_ranking(scores, per_head):
    ranked = sorted(per_head, key=lambda x: x["rel"])
    print(f"\n  {'Rank':<6}  {'Layer':^5}  {'Head':^5}  "
          f"{'Abl logit-diff':>16}  {'Rel change':>12}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*16}  {'─'*12}")
    for rank, d in enumerate(ranked, 1):
        print(f"  {rank:<6}  L{d['layer']:<4}  H{d['head']:<4}  "
              f"{d['abl_ld']:>16.4f}  {d['rel']:>+12.4f}")


def print_by_layer(scores, relevant_set, threshold):
    n_l, n_h = scores.shape
    print(f"\n  threshold = {threshold}  "
          f"(heads with score < threshold are KEPT in circuit)\n")
    for l in range(n_l):
        kept    = [(h, scores[l, h].item()) for h in range(n_h)
                   if (l, h) in relevant_set]
        ablated = [h for h in range(n_h) if (l, h) not in relevant_set]
        kept_s  = "  ".join(f"H{h}({s:+.3f})" for h, s in kept) or "—"
        abl_s   = " ".join(f"H{h}" for h in ablated) or "—"
        print(f"  L{l:<2}  KEPT ({len(kept):2d}): {kept_s}")
        print(f"       ABL  ({len(ablated):2d}): {abl_s}")
        print()


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
        rank_s = str(r["io_rank"]) if r["io_rank"] else f">{top_n}"
        top_str = "  ".join(
            f"'{r['top_tokens'][k].strip()}' {r['top_probs'][k]*100:.1f}%"
            for k in range(min(top_n, len(r["top_tokens"]))))
        print(f"  {r['idx']+1:<6}  {r['io']:^6}  {r['s']:^6}  "
              f"{r['logit_diff']:>7.3f}  {rank_s:^5}  {top_str}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule()
print("  Biased IOI — EasyAblation Study + Logit Inspection")
rule()
print(f"""
  numb={numb}  bias={bias_ratio:.0%} {NAME_MAJORITY}=IO  switch={switch_perc:.0%}  seed={seed}
  top_n={top_n}  batch_size={batch_size}  threshold={ABLATION_THRESHOLD}
  ablation strategy: EasyAblation  abl_type=mean  head_circuit=result
  device={device}
""")
rule()

# ── 1. Dataset ────────────────────────────────────────────────────────────────
section("STEP 1 — BUILD DATASET")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = BiasedIOIDataset(
    N=numb, switch_perc=switch_perc, bias_ratio=bias_ratio,
    tokenizer=tokenizer, prepend_bos=True, seed=seed,
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

# use_attn_result = True is REQUIRED for head_circuit="result" in EasyAblation.
# It registers hook_result hooks (head output after W_O projection) in addition
# to the default hook_z (head output before W_O projection).
model = EasyTransformer.from_pretrained("gpt2")
model.cfg.use_attn_result = True
model.to(device)
model.eval()
print(f"\n  GPT-2 small: {model.cfg.n_layers} layers × "
      f"{model.cfg.n_heads} heads × d_model={model.cfg.d_model}")
print(f"  Total heads: {model.cfg.n_layers * model.cfg.n_heads}")
print(f"  use_attn_result = True  (hook_result hooks are active)")
print(f"  Device: {device}")

# ── 3. Semantic indices from dataset ─────────────────────────────────────────
section("STEP 3 — BUILD SEMANTIC INDICES")
print(f"""
  semantic_indices maps semantic labels to the per-prompt token position of
  each named entity.  EasyAblation uses these so that when computing the
  "mean" activation to use as replacement, it averages each head's output
  separately at IO positions, S positions, etc. — rather than using one
  global mean that blurs across positions.

  Positions are extracted from dataset.word_idx.
""")

def to_list(val):
    if val is None: return None
    if hasattr(val, 'tolist'): return val.tolist()
    return list(val)

semantic_indices = {}
for key in ["IO", "S", "S2", "end"]:
    if key in dataset.word_idx:
        semantic_indices[key] = to_list(dataset.word_idx[key])
    else:
        print(f"  WARNING: '{key}' not found in dataset.word_idx — skipping.")

print(f"  Keys found: {list(semantic_indices.keys())}")
for key, vals in semantic_indices.items():
    uniq = sorted(set(vals))
    print(f"  '{key}':  {len(vals)} entries,  "
          f"unique positions = {uniq[:10]}{'...' if len(uniq)>10 else ''}")

# Also collect the flat lists needed inside logit_diff
ioi_text_prompts = dataset.sentences
ioi_end_idx      = to_list(dataset.word_idx["end"])
ioi_io_ids       = list(dataset.io_tokenIDs)   # token ID of IO name per prompt
ioi_s_ids        = list(dataset.s_tokenIDs)    # token ID of S  name per prompt

# Sanity-check: decode a few IO/S token IDs
print(f"\n  Token ID sanity check (first 5 prompts):")
print(f"  {'#':<5}  {'IO name':^8}  {'IO token ID':^12}  "
      f"{'IO decoded':^12}  {'S name':^8}  {'S token ID':^12}  {'S decoded':^12}")
print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*12}")
for i in range(min(5, numb)):
    p      = dataset.ioi_prompts[i]
    io_dec = tokenizer.decode(ioi_io_ids[i])
    s_dec  = tokenizer.decode(ioi_s_ids[i])
    print(f"  {i+1:<5}  {p['IO']:^8}  {ioi_io_ids[i]:^12}  "
          f"{io_dec!r:^12}  {p['S']:^8}  {ioi_s_ids[i]:^12}  {s_dec!r:^12}")

# ── 4. EasyAblation ───────────────────────────────────────────────────────────
section("STEP 4 — EASY ABLATION  (abl_type=mean, head_circuit=result)")
print(f"""
  For each of the {model.cfg.n_layers * model.cfg.n_heads} heads:
    1. Patch hook_result so that head (L, H) always outputs its global mean
       activation (cached from the first {ABLATION_MEAN_N} prompts).
    2. Run ALL {numb} prompts through the patched model (batched, in logit_diff).
    3. Compute mean logit-diff (IO − S) at the prediction position.
    4. Score = (ablated_ld − baseline_ld) / |baseline_ld|   (relative_metric=True)

  More negative score → ablating this head hurts IOI more → more important.

  NOTE: semantic_indices is not passed to EasyAblation — its implementation
  issues a warning for large datasets and causes extra memory overhead.
  Global means (one mean vector per head) are used instead.
""", flush=True)

# logit_diff is called by EasyAblation with (model, ioi_text_prompts).
# We use pre-tokenized tensors (dataset.toks) for speed and correctness,
# ignoring the string list argument since it is always the same dataset.
_toks    = dataset.toks
_end_idx = ioi_end_idx
_io_ids  = ioi_io_ids
_s_ids   = ioi_s_ids
_N       = numb

def logit_diff(model, _text_prompts):
    """
    Batched logit-diff for EasyAblation.
    Uses pre-tokenized dataset.toks rather than re-tokenizing the strings,
    which is both faster and avoids any tokenizer padding differences.
    Called once per head ablation (144+ times total).
    """
    all_diffs = []
    for i in range(0, _N, batch_size):
        end_b  = min(i + batch_size, _N)
        batch  = _toks[i:end_b].to(device)
        b_end  = _end_idx[i:end_b]
        b_io   = _io_ids[i:end_b]
        b_s    = _s_ids[i:end_b]
        with torch.no_grad():
            logits = model(batch).detach().cpu()
        for j in range(end_b - i):
            all_diffs.append(
                logits[j, b_end[j], b_io[j]].item() -
                logits[j, b_end[j], b_s[j]].item()
            )
    return torch.tensor(sum(all_diffs) / len(all_diffs))

# The metric uses ALL numb prompts (full dataset) for accurate logit-diff scores.
metric = ExperimentMetric(
    metric=logit_diff,
    dataset=ioi_text_prompts,
    relative_metric=True,
)

# mean_dataset is capped at ABLATION_MEAN_N because EasyAblation's get_all_mean()
# runs model(mean_dataset) as a SINGLE forward pass — too many strings causes OOM.
# The metric itself (logit_diff) still uses the full dataset via batching.
#
# semantic_indices is NOT passed: EasyAblation's implementation issues a warning
# when it receives per-prompt position lists, and with large datasets it causes
# additional memory overhead.  Global means (no positional differentiation) are
# used instead — this is the standard mean-ablation approach.
mean_subset = ioi_text_prompts[:ABLATION_MEAN_N]

config = AblationConfig(
    abl_type="mean",
    mean_dataset=mean_subset,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=True,
)

print(f"  mean_dataset size for EasyAblation: {len(mean_subset)} prompts "
      f"(capped from {numb} to avoid OOM in get_all_mean)",
      file=sys.stderr, flush=True)

abl    = EasyAblation(model, config, metric)   # no semantic_indices
print("  Running EasyAblation.run_ablation() ...", file=sys.stderr, flush=True)
result = abl.run_ablation()   # [n_layers, n_heads] tensor

# EasyAblation does not expose the baseline directly, but logit_diff(model, …)
# with no hooks gives us the baseline.
print("  Computing baseline logit-diff (no ablation) ...",
      file=sys.stderr, flush=True)
with torch.no_grad():
    baseline_ld = logit_diff(model, ioi_text_prompts).item()

# ── 4a. Score matrix ──────────────────────────────────────────────────────────
section("STEP 4a — SCORE MATRIX  [layer × head]")
print_score_matrix(result, model.cfg.n_layers, model.cfg.n_heads)
print(f"\n  Baseline (unablated) mean logit-diff: {baseline_ld:.4f}")

# Build per_head list for ranking output (mirrors compute_ablation_scores output)
per_head = []
for l in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        rel = result[l, h].item()
        # recover approximate ablated logit-diff from relative score
        abl_ld = baseline_ld * (1 + rel) if baseline_ld != 0 else rel
        per_head.append(dict(layer=l, head=h, abl_ld=abl_ld, rel=rel))

# ── 4b. Ranked list ───────────────────────────────────────────────────────────
section("STEP 4b — ALL HEADS RANKED BY IMPORTANCE  (most → least harmful to ablate)")
print_head_ranking(result, per_head)

# ── 4c. Circuit selection ─────────────────────────────────────────────────────
section(f"STEP 4c — CIRCUIT SELECTION  (threshold = {ABLATION_THRESHOLD})")
relevant_heads = select_circuit_heads(result, ABLATION_THRESHOLD)
relevant_set   = set(relevant_heads)
print_by_layer(result, relevant_set, ABLATION_THRESHOLD)
print(f"  TOTAL: {len(relevant_heads)} heads kept in circuit  |  "
      f"{model.cfg.n_layers * model.cfg.n_heads - len(relevant_heads)} mean-ablated")

# ── 5. Head means for circuit hooks ──────────────────────────────────────────
section("STEP 5 — COMPUTE HEAD MEANS FOR CIRCUIT HOOKS")
print(f"""
  EasyAblation caches means internally for its own ablation passes.
  Here we recompute hook_result means over the full dataset so we can build
  our own persistent hooks for the logit inspection step below.
  (hook_result shape per head: d_model = {model.cfg.d_model})
""", flush=True)

head_means = compute_head_means(model, dataset.toks, batch_size, device)
print(f"\n  Done.  head_means shape: {list(head_means.shape)}")
print(f"  Mean L2 norm per head, first 3 layers (sanity check):")
for l in range(min(3, model.cfg.n_layers)):
    norms = head_means[l].norm(dim=-1)
    print(f"    L{l}: " + "  ".join(f"H{h}={norms[h]:.3f}" for h in range(model.cfg.n_heads)))

ablation_hooks = build_ablation_hooks(model, relevant_set, head_means, device)

# ── 6. Logit inspection — Full model ─────────────────────────────────────────
section(f"STEP 6 — LOGIT INSPECTION: FULL GPT-2  "
        f"(all {model.cfg.n_layers * model.cfg.n_heads} heads active)")
recs_full = collect_top_n(model, dataset, top_n, batch_size, device,
                           desc="full model")
print_logit_table(recs_full, baseline_ld=None)

# ── 7. Logit inspection — Circuit-only ───────────────────────────────────────
section(f"STEP 7 — LOGIT INSPECTION: CIRCUIT-ONLY  ({len(relevant_heads)} heads active)")
print(f"\n  {model.cfg.n_layers * model.cfg.n_heads - len(relevant_heads)} non-circuit "
      f"heads have their hook_result replaced with their dataset mean.")
recs_circuit = collect_top_n(model, dataset, top_n, batch_size, device,
                               ablation_hooks=ablation_hooks,
                               desc="circuit-only")
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

for name, recs in [(f"Full GPT-2", recs_full),
                   (f"Circuit-only ({len(relevant_heads)} heads)", recs_circuit)]:
    maj_ld = [r["logit_diff"] for r in recs if r["io"] == NAME_MAJORITY]
    min_ld = [r["logit_diff"] for r in recs if r["io"] == NAME_MINORITY]
    maj_r1 = sum(1 for r in recs if r["io"] == NAME_MAJORITY and r["io_rank"] == 1)
    min_r1 = sum(1 for r in recs if r["io"] == NAME_MINORITY and r["io_rank"] == 1)

    print(f"\n  {name}")
    print(f"  {'─'*62}")
    print(f"  IO={NAME_MAJORITY} ({len(maj_ld)} prompts):"
          f"  mean LD = {sum(maj_ld)/len(maj_ld):+.4f}"
          f"  |  rank-1 accuracy = {maj_r1}/{len(maj_ld)} "
          f"({100*maj_r1/len(maj_ld):.1f}%)")
    print(f"  IO={NAME_MINORITY} ({len(min_ld)} prompts):"
          f"  mean LD = {sum(min_ld)/len(min_ld):+.4f}"
          f"  |  rank-1 accuracy = {min_r1}/{len(min_ld)} "
          f"({100*min_r1/len(min_ld):.1f}%)")
    overall_r1 = maj_r1 + min_r1
    overall_ld = (sum(maj_ld) + sum(min_ld)) / (len(maj_ld) + len(min_ld))
    print(f"  Overall:                "
          f"  mean LD = {overall_ld:+.4f}"
          f"  |  rank-1 accuracy = {overall_r1}/{numb} "
          f"({100*overall_r1/numb:.1f}%)")

print()
rule()
print("  Done.")
rule()
