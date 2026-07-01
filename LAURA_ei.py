"""
LAURA_ei.py

Computes Effective Information (EI) and Effectiveness (Eff) for both the
micro and macro descriptions of the GPT-2 IOI predictions, following
Hoel et al. (2013).

Causal model
------------
  Causes  : {Mary=IO, John=IO}        — n_causes = 2
  UC      : uniform = [0.5, 0.5]      — equal probability of each prompt type

  Micro effects : renormalised top-N token distribution  (K = top_n states)
  Macro effects : {Mary, John, Other}                     (3 states)

  TPM[cause_i] = average distribution over all prompts of type cause_i

EI formula (Hoel et al. eq.)
-----------------------------
  EI = H(UE) - (1/n_causes) * Σ_causes H(effect | cause)

  where UE = marginal effect under uniform cause = average of the two rows of TPM

  H = Shannon entropy in bits = -Σ p * log2(p)

Normalised effectiveness
------------------------
  Eff = EI / log2(n_effect_states)
      = 1  for a perfect (deterministic, non-degenerate) system
      = 0  for completely noisy or completely degenerate

Causal emergence
----------------
  CE_raw  = EI_macro  - EI_micro     (raw bits — different scales, use with caution)
  CE_eff  = Eff_macro - Eff_micro    (normalised — the meaningful comparison)

  CE_eff > 0  →  macro description is causally more structured than micro

HOW TO RUN
----------
    python LAURA_ei.py > ei_results.out

Input: biased_logits_results.pt  (same file used by LAURA_macro_states.py)
"""

import os
import sys
import math
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────

input_file = "biased_logits_results.pt"

# ──────────────────────────────────────────────────────────────────────────────

if not os.path.exists(input_file):
    sys.exit(f"ERROR: '{input_file}' not found. Run LAURA_biased_logits.py first.")

data        = torch.load(input_file, map_location="cpu")
top_probs   = data["top_probs"]    # [N, top_n]  already renormalised to sum=1
top_tokens  = data["top_tokens"]   # list[list[str]]
io_names    = data["io_names"]     # list[str]
cfg         = data["config"]

N     = len(io_names)
top_n = cfg["top_n"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def entropy(dist):
    """Shannon entropy in bits of a 1-D array-like that sums to 1."""
    h = 0.0
    for p in dist:
        if p > 0:
            h -= p * math.log2(p)
    return h


def average_dist(dists):
    """Element-wise average of a list of equal-length lists."""
    n = len(dists)
    k = len(dists[0])
    avg = [0.0] * k
    for d in dists:
        for j in range(k):
            avg[j] += d[j] / n
    return avg


def compute_ei(dist_cause_A, dist_cause_B):
    """
    EI for a 2-cause system under uniform intervention.
    dist_cause_A, dist_cause_B : lists of probabilities over effect states (sum to 1)
    Returns: EI (bits), Eff, H_UE, H_A, H_B
    """
    n_causes     = 2
    n_effects    = len(dist_cause_A)

    # marginal effect under uniform cause distribution
    ue           = [(a + b) / 2 for a, b in zip(dist_cause_A, dist_cause_B)]

    H_UE  = entropy(ue)
    H_A   = entropy(dist_cause_A)
    H_B   = entropy(dist_cause_B)
    avg_H = (H_A + H_B) / n_causes

    EI    = max(H_UE - avg_H, 0.0)          # clamp to 0 (can be tiny negative due to float)
    Eff   = EI / math.log2(n_effects) if n_effects > 1 else 0.0

    return EI, Eff, H_UE, H_A, H_B


# ── BUILD TPM ROWS ────────────────────────────────────────────────────────────
# For each group, collect per-prompt distributions then average → one TPM row.

# ── MICRO: average renormalised top-N distributions ──────────────────────────

micro_mary = []   # top-N distribution for each Mary=IO prompt
micro_john = []   # top-N distribution for each John=IO prompt

for i in range(N):
    dist = top_probs[i].tolist()    # list of top_n floats, sums to 1
    if io_names[i] == "Mary":
        micro_mary.append(dist)
    else:
        micro_john.append(dist)

tpm_micro_mary = average_dist(micro_mary)   # [top_n]
tpm_micro_john = average_dist(micro_john)   # [top_n]

# ── MACRO: collapse top-N into {Mary, John, Other} ───────────────────────────

def macro_dist(top_probs_row, top_tokens_row):
    """Map one prompt's top-N distribution to (p_mary, p_john, p_other)."""
    p_mary = p_john = 0.0
    for k, tok in enumerate(top_tokens_row):
        p = top_probs_row[k]
        if tok.strip() == "Mary":
            p_mary += p
        elif tok.strip() == "John":
            p_john += p
    p_other = max(0.0, 1.0 - p_mary - p_john)
    return [p_mary, p_john, p_other]

macro_mary = []
macro_john = []

for i in range(N):
    d = macro_dist(top_probs[i].tolist(), top_tokens[i])
    if io_names[i] == "Mary":
        macro_mary.append(d)
    else:
        macro_john.append(d)

tpm_macro_mary = average_dist(macro_mary)   # [p_mary, p_john, p_other]
tpm_macro_john = average_dist(macro_john)   # [p_mary, p_john, p_other]

# ── COMPUTE EI ────────────────────────────────────────────────────────────────

EI_micro,  Eff_micro,  H_UE_micro,  H_micro_mary,  H_micro_john  = \
    compute_ei(tpm_micro_mary, tpm_micro_john)

EI_macro,  Eff_macro,  H_UE_macro,  H_macro_mary,  H_macro_john  = \
    compute_ei(tpm_macro_mary, tpm_macro_john)

CE_raw = EI_macro  - EI_micro
CE_eff = Eff_macro - Eff_micro

n_mary = len(micro_mary)
n_john = len(micro_john)

# ── OUTPUT ────────────────────────────────────────────────────────────────────

W = 66

def rule(c="═"):
    print(c * W)

rule()
print(f"  Effective Information — Hoel et al. (2013) framework")
print(f"  N={N}  |  top_n={top_n}  |  causes: {{Mary=IO, John=IO}}")
rule()

print(f"""
  Causal model
  ─────────────────────────────────────────────────────────
  Cause states   : {{Mary=IO (n={n_mary}), John=IO (n={n_john})}}
  Intervention   : uniform — UC = [0.5, 0.5]
  Micro effects  : {top_n} renormalised tokens  (max EI = log2({top_n}) = {math.log2(top_n):.4f} bits)
  Macro effects  : {{Mary, John, Other}}          (max EI = log2(3)   = {math.log2(3):.4f} bits)
""")

# ── TPM rows ──────────────────────────────────────────────────────────────────

rule("─")
print("  TRANSITION PROBABILITY MATRIX  (averaged over prompts per group)")
rule("─")

print(f"\n  MICRO  (top-{top_n} tokens, showing top-5 for readability)")
top5_names_mary = [top_tokens[next(i for i in range(N) if io_names[i]=="Mary")][k] for k in range(5)]
print(f"  Mary=IO row  :  " + "  ".join(f"{v*100:5.2f}%" for v in tpm_micro_mary[:5]) + "  ...")
print(f"  John=IO row  :  " + "  ".join(f"{v*100:5.2f}%" for v in tpm_micro_john[:5]) + "  ...")

print(f"\n  MACRO  ({{Mary, John, Other}})")
print(f"  {'Group':<14}  {'P(Mary)':>9}  {'P(John)':>9}  {'P(Other)':>9}")
print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*9}")
print(f"  {'Mary=IO':<14}  {tpm_macro_mary[0]*100:>8.3f}%  {tpm_macro_mary[1]*100:>8.3f}%  {tpm_macro_mary[2]*100:>8.3f}%")
print(f"  {'John=IO':<14}  {tpm_macro_john[0]*100:>8.3f}%  {tpm_macro_john[1]*100:>8.3f}%  {tpm_macro_john[2]*100:>8.3f}%")

# ── Entropy components ────────────────────────────────────────────────────────

rule("─")
print("  ENTROPY COMPONENTS  (bits)")
rule("─")
print(f"""
  {'':30}  {'MICRO':>10}  {'MACRO':>10}
  {'H(effect | Mary=IO)':<30}  {H_micro_mary:>10.4f}  {H_macro_mary:>10.4f}
  {'H(effect | John=IO)':<30}  {H_micro_john:>10.4f}  {H_macro_john:>10.4f}
  {'Avg H(effect | cause)':<30}  {(H_micro_mary+H_micro_john)/2:>10.4f}  {(H_macro_mary+H_macro_john)/2:>10.4f}
  {'H(UE)  [marginal effect]':<30}  {H_UE_micro:>10.4f}  {H_UE_macro:>10.4f}
""")

# ── Main results ──────────────────────────────────────────────────────────────

rule("═")
print("  RESULTS")
rule("═")
print(f"""
  {'Metric':<35}  {'MICRO':>10}  {'MACRO':>10}
  {'─'*35}  {'─'*10}  {'─'*10}
  {'EI  (bits)':<35}  {EI_micro:>10.4f}  {EI_macro:>10.4f}
  {'max EI = log2(n_states)  (bits)':<35}  {math.log2(top_n):>10.4f}  {math.log2(3):>10.4f}
  {'Eff = EI / log2(n_states)':<35}  {Eff_micro:>10.4f}  {Eff_macro:>10.4f}
  {'─'*35}  {'─'*10}  {'─'*10}
  {'ΔEI  (macro − micro, raw bits)':<35}  {CE_raw:>+10.4f}
  {'ΔEff (macro − micro, normalised)':<35}  {CE_eff:>+10.4f}
""")

rule("─")
if CE_eff > 0:
    print(f"  ΔEff = {CE_eff:+.4f}  →  CAUSAL EMERGENCE")
    print(f"  The macro description ({{Mary, John, Other}}) is more causally")
    print(f"  structured than the micro top-{top_n} token description.")
elif CE_eff < 0:
    print(f"  ΔEff = {CE_eff:+.4f}  →  CAUSAL REDUCTION")
    print(f"  The micro top-{top_n} token description is more causally")
    print(f"  structured than the macro description.")
else:
    print(f"  ΔEff = 0.0000  →  equal causal structure at both levels.")
rule("─")
print()
