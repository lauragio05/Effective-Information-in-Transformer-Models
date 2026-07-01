import torch
import matplotlib
matplotlib.use("Agg")   # headless server — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from easy_transformer import EasyTransformer
from easy_transformer.experiments import EasyAblation, AblationConfig, ExperimentMetric

ioi_text_prompts = [
    "Then, Christina and Samantha were working at the grocery store. Samantha decided to give a kiss to Christina",
    "Then, Samantha and Christina were working at the grocery store. Christina decided to give a kiss to Samantha",
    "When Timothy and Dustin got a kiss at the grocery store, Dustin decided to give it to Timothy",
    "When Dustin and Timothy got a kiss at the grocery store, Timothy decided to give it to Dustin",
]
ioi_io_ids = [33673, 34778, 22283, 37616]
ioi_s_ids  = [34778, 33673, 37616, 22283]
ioi_end_idx = [18, 18, 17, 17]
semantic_indices = {
    "IO":  [2, 2, 1, 1],
    "S":   [4, 4, 3, 3],
    "S2":  [12, 12, 12, 12],
    "end": ioi_end_idx,
}

def logit_diff(model, text_prompts):
    logits = model(text_prompts).detach()
    IO_logits = logits[torch.arange(len(text_prompts)), ioi_end_idx, ioi_io_ids]
    S_logits  = logits[torch.arange(len(text_prompts)), ioi_end_idx, ioi_s_ids]
    return (IO_logits - S_logits).mean().detach().cpu()

# Load model
model = EasyTransformer.from_pretrained("gpt2")
model.cfg.use_attn_result = True
if torch.cuda.is_available():
    model.to("cuda")

# Run ablation
metric = ExperimentMetric(metric=logit_diff, dataset=ioi_text_prompts, relative_metric=True)
config = AblationConfig(
    abl_type="mean",
    mean_dataset=ioi_text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=True,
)
abl = EasyAblation(model, config, metric, semantic_indices=semantic_indices)
result = abl.run_ablation()

# Print raw numbers
print("\n=== Ablation Results (layer x head) ===")
print(result)

# Find the most important heads (most negative = most harmful to ablate)
print("\n=== Top 10 most important heads ===")
flat = result.flatten()
top_indices = flat.argsort()[:10]
for idx in top_indices:
    layer = idx.item() // model.cfg.n_heads
    head  = idx.item() %  model.cfg.n_heads
    print(f"  Layer {layer}, Head {head}: {result[layer][head]:.4f}")

# Heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(
    result.numpy(),
    annot=True,
    fmt=".2f",
    cmap="RdBu",
    center=0,
    xticklabels=[f"H{i}" for i in range(model.cfg.n_heads)],
    yticklabels=[f"L{i}" for i in range(model.cfg.n_layers)],
)
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Semantic Ablation — Relative logit diff change per head\n(negative = head is important for IOI task)")
plt.tight_layout()
plt.savefig("ablation_results.png", dpi=150)
print("\nHeatmap saved to ablation_results.png")