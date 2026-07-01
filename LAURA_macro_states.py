"""
LAURA_macro_states.py

Loads the saved results from LAURA_biased_logits.py and collapses the
top-N token predictions into three macro states:

    Mary   – probability mass on the token ' Mary'
    John   – probability mass on the token ' John'
    Other  – everything else in the top-N  (1 - P(Mary) - P(John))

The top-N probabilities are already normalised to sum to 1 by
LAURA_biased_logits.py at save time, so no renormalisation is needed here.

The averages are computed separately for:
    • prompts where Mary is the IO  (correct answer = Mary)
    • prompts where John is the IO  (correct answer = John)

HOW TO RUN
----------
    python LAURA_macro_states.py > macro_states.out

Input file  : biased_logits_results.pt   (same directory, created by LAURA_biased_logits.py)
Output file : macro_states.out           (redirect stdout as above)
"""

import os
import sys
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────

input_file = "biased_logits_results.pt"   # produced by LAURA_biased_logits.py

# ──────────────────────────────────────────────────────────────────────────────

# ── LOAD ──────────────────────────────────────────────────────────────────────

if not os.path.exists(input_file):
    sys.exit(f"ERROR: '{input_file}' not found. Run LAURA_biased_logits.py first.")

data = torch.load(input_file, map_location="cpu")

top_probs   = data["top_probs"]     # [N, top_n]  true softmax probs over full vocab
top_tokens  = data["top_tokens"]    # list[list[str]]
io_names    = data["io_names"]      # list[str]
s_names     = data["s_names"]       # list[str]
prompts     = data["prompts"]       # list[str]
cfg         = data["config"]

N     = len(prompts)
top_n = cfg["top_n"]

# ── PER-PROMPT MACRO STATES ───────────────────────────────────────────────────
# For each prompt:
#   1. Re-normalise the top-N probs so they sum to 1.
#   2. Assign each of the top-N tokens to Mary / John / Other.
#   3. Sum within each macro state.

per_prompt = []   # list of dicts {io, s, prompt, p_mary, p_john, p_other}

for i in range(N):
    probs = top_probs[i]                  # [top_n]  already sums to 1 (normalised in LAURA_biased_logits.py)

    p_mary  = 0.0
    p_john  = 0.0

    for k in range(top_n):
        tok   = top_tokens[i][k].strip()  # strip leading space GPT-2 adds
        p_val = probs[k].item()
        if tok == "Mary":
            p_mary += p_val
        elif tok == "John":
            p_john += p_val
        # anything else goes to Other implicitly

    p_other = max(0.0, 1.0 - p_mary - p_john)   # remainder

    per_prompt.append({
        "io":      io_names[i],
        "s":       s_names[i],
        "prompt":  prompts[i],
        "p_mary":  p_mary,
        "p_john":  p_john,
        "p_other": p_other,
    })

# ── GROUP AVERAGES ────────────────────────────────────────────────────────────

def group_avg(group_io):
    subset = [p for p in per_prompt if p["io"] == group_io]
    if not subset:
        return None, None, None, 0
    avg_mary  = sum(p["p_mary"]  for p in subset) / len(subset)
    avg_john  = sum(p["p_john"]  for p in subset) / len(subset)
    avg_other = sum(p["p_other"] for p in subset) / len(subset)
    return avg_mary, avg_john, avg_other, len(subset)

mary_avg_mary,  mary_avg_john,  mary_avg_other,  n_mary = group_avg("Mary")
john_avg_mary,  john_avg_john,  john_avg_other,  n_john = group_avg("John")

# ── FORMATTED OUTPUT ──────────────────────────────────────────────────────────

W = 72

def rule(c="═"):
    print(c * W)

def section(title):
    print()
    rule("─")
    print(f"  {title}")
    rule("─")

rule()
print(f"  Macro-State Analysis  |  top-{top_n} re-normalised  |  N={N}")
print(f"  Source: {input_file}")
rule()

# ── Per-prompt listing ────────────────────────────────────────────────────────

for group_io, group_s in [("Mary", "John"), ("John", "Mary")]:
    subset = [p for p in per_prompt if p["io"] == group_io]
    section(f"PROMPTS WHERE  IO={group_io}  S={group_s}  ({len(subset)} prompts)")

    print(f"\n  {'#':<5}  {'Mary':>8}  {'John':>8}  {'Other':>8}   Prompt")
    print(f"  {'─'*5}  {'────────':>8}  {'────────':>8}  {'────────':>8}   {'─'*40}")

    for j, p in enumerate(subset, 1):
        short = p["prompt"][:60] + ("…" if len(p["prompt"]) > 60 else "")
        print(
            f"  {j:<5}  "
            f"{p['p_mary']*100:>7.2f}%  "
            f"{p['p_john']*100:>7.2f}%  "
            f"{p['p_other']*100:>7.2f}%   "
            f"{short}"
        )

# ── Summary table ─────────────────────────────────────────────────────────────

section("AVERAGES ACROSS ALL PROMPTS")

print(f"""
  ┌──────────────────────┬──────────────┬──────────────┬──────────────┐
  │  Group               │  P( Mary )   │  P( John )   │  P( Other )  │
  ├──────────────────────┼──────────────┼──────────────┼──────────────┤
  │  IO=Mary  (n={n_mary:<4})  │  {mary_avg_mary*100:>8.2f}%  │  {mary_avg_john*100:>8.2f}%  │  {mary_avg_other*100:>8.2f}%  │
  │  IO=John  (n={n_john:<4})  │  {john_avg_mary*100:>8.2f}%  │  {john_avg_john*100:>8.2f}%  │  {john_avg_other*100:>8.2f}%  │
  └──────────────────────┴──────────────┴──────────────┴──────────────┘

  The correct IO token is marked with ◄
  IO=Mary group:  Mary {mary_avg_mary*100:.2f}% ◄   John {mary_avg_john*100:.2f}%   Other {mary_avg_other*100:.2f}%
  IO=John group:  Mary {john_avg_mary*100:.2f}%    John {john_avg_john*100:.2f}% ◄  Other {john_avg_other*100:.2f}%
""")

rule()
