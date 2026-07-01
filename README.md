# Effective Information in Transformer Models

Master's thesis code — UPC Barcelona, 2024–2025

Applies the **Effective Information (EI)** framework (Hoel et al., 2013) to study **causal emergence** in GPT-2 Small's Indirect Object Identification (IOI) circuit (Wang et al., 2022). The central question: does the macro-level description — collapsing the token distribution to `{Mary, John, Other}` — carry *more* causal structure than the full micro-level token distribution?

---

## Setup

### Step 1 — Clone and install Easy-Transformer

This code depends on the Easy-Transformer library. Clone it first and install its requirements:

```bash
git clone https://github.com/redwoodresearch/Easy-Transformer/
pip install -r requirements.txt
```

### Step 2 — Add the two dataset files to Easy-Transformer

After cloning Easy-Transformer, copy the two files from this repo's `easy_transformer/` folder into `Easy-Transformer/easy_transformer/`:

```
Easy-Transformer/
└── easy_transformer/
    ├── LAURA_biased_ioi_dataset.py       ← put this here
    └── LAURA_biased_ioi_dataset_extra.py ← put this here
```

### Step 3 — Set the path in each script

At the top of each `LAURA_*.py` script, set `EASY_TRANSFORMER_DIR` to the path where you cloned Easy-Transformer:

```python
EASY_TRANSFORMER_DIR = "/path/to/Easy-Transformer"
```

---

## Files in this repo

### Root scripts

| Script | What it does |
|---|---|
| `LAURA_full_experiment.py` | **Start here.** Runs the complete pipeline (logits → macro states → EI) for three conditions: Full GPT-2, Circuit-only, Untrained GPT-2 |
| `LAURA_full_experiment_extra.py` | Extended version of the above |
| `LAURA_biased_logits.py` | Forward-pass baseline. Records top-N next-token predictions on a biased IOI dataset (70% Mary=IO / 30% John=IO) |
| `LAURA_macro_states.py` | Collapses top-N predictions to macro states `{Mary, John, Other}` |
| `LAURA_ei.py` | Computes EI, Effectiveness (Eff), and ΔEff (causal emergence) at micro and macro levels |
| `LAURA_circuit_logits.py` | Logit inspection using the hardcoded 26-head IOI circuit with mean-ablation of all other heads |
| `LAURA_ablation_study.py` | Full ablation study — score matrix `[n_layers × n_heads]`, circuit head selection, full vs. circuit-only comparison |
| `LAURA_run_ablation.py` | Runs the ablation pipeline |
| `LAURA_collect_head_datasets.py` | Collects per-attention-head datasets for downstream analysis |

### Files to place in Easy-Transformer

| File | What it does |
|---|---|
| `easy_transformer/LAURA_biased_ioi_dataset.py` | `BiasedIOIDataset`: generates IOI prompts with a controllable Mary/John frequency bias |
| `easy_transformer/LAURA_biased_ioi_dataset_extra.py` | Extended dataset with additional controls |

---

## Running the experiments

### Full pipeline (recommended)

```bash
python LAURA_full_experiment.py > full_experiment.out 2>progress.err
```

Runs all three conditions on a shared biased dataset and prints EI / Eff / ΔEff for each.

### Step-by-step

```bash
python LAURA_biased_logits.py > results.out 2>progress.err
python LAURA_macro_states.py  > macro_states.out
python LAURA_ei.py            > ei_results.out
```

### Ablation study

```bash
python LAURA_ablation_study.py  > ablation.out 2>progress.err
python LAURA_circuit_logits.py  > circuit_logits.out 2>progress.err
```

---

## Key concepts

**Effective Information (EI)** — measures how much a cause constrains its effects under uniform intervention.

**Effectiveness (Eff)** — normalised EI: `EI / log2(n_states)`. Ranges 0–1.

**Causal emergence (ΔEff)** — `Eff_macro − Eff_micro`. Positive means the coarse-grained `{Mary, John, Other}` description is *more* causally structured than the full token distribution.

**IOI task** — GPT-2 is given sentences like *"When Mary and John went to the store, John gave a drink to ___"* and asked to predict the next token (correct answer: Mary). The IOI circuit identifies the 26 attention heads responsible.

---

## References

- Hoel, E. P., Albantakis, L., & Tononi, G. (2013). Quantifying causal emergence shows that macro can beat micro. *PNAS*, 110(49), 19790–19795.
- Wang, K. R., et al. (2022). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *arXiv:2211.00593*.
