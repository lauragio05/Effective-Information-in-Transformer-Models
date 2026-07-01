"""
collect_head_datasets.py

Collects input/output activation tensors for two attention heads:
  - IOI_HEAD:  the most important head for the IOI task  (L11 H11, ablation score -0.43)
  - CTRL_HEAD: a head unimportant for IOI but still active on ABC prompts (L1 H6)

Runs both heads on both datasets (IOI prompts and ABC prompts).
Produces 4 output files:
  - activations_IOI_head_IOI_promptsBIG.pt
  - activations_IOI_head_ABC_promptsBIG.pt
  - activations_CTRL_head_IOI_promptsBIG.pt
  - activations_CTRL_head_ABC_promptsBIG.pt

Each file contains a dict:
  {
    "inputs":  tensor [numb, seq_len, 64],
    "outputs": tensor [numb, seq_len, 64],
    "prompts": list of strings (the raw text)
  }

Change `numb` to control how many prompts are used per dataset.
"""

import torch
import sys
import os
import gc
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easy_transformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from transformers import AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────

numb = 10000  # number of prompts to use for each dataset

IOI_HEAD  = (11, 11)  # ablation score -0.43, most important for IOI
CTRL_HEAD = (1,  6)   # near-zero IOI score (~0.005), candidate control head

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

print("Loading model...")
model = EasyTransformer.from_pretrained("gpt2")
# IMPORTANT: leave use_attn_result = False (the default).
# Setting it True materializes a [batch, seq, n_heads, d_model] tensor at every
# layer's hook_result, which we don't need and which blows up GPU memory.
model.cfg.use_attn_result = False
if torch.cuda.is_available():
    model.to("cuda")
model.eval()

d_head = model.cfg.d_head  # 64 for GPT-2 small

# ── BUILD DATASETS ────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(f"\nGenerating {numb} IOI prompts...")
ioi_dataset = IOIDataset(
    prompt_type="BABA",
    N=numb,
    tokenizer=tokenizer,
    prepend_bos=True,
)
ioi_prompts = ioi_dataset.sentences

print(f"Generating {numb} ABC prompts...")
abc_dataset = IOIDataset(
    prompt_type="ABC",
    N=numb,
    tokenizer=tokenizer,
    prepend_bos=True,
)
abc_prompts = abc_dataset.sentences

print(f"\nExample IOI prompt:  {ioi_prompts[0]}")
print(f"Example ABC prompt:  {abc_prompts[0]}")

# ── ACTIVATION COLLECTION ─────────────────────────────────────────────────────

def collect_activations(model, prompts, layer, head, desc=""):
    """
    For each prompt, collect:
      input:  residual stream slice for this head  [1, seq_len, 64]
              (NOTE: the residual stream is NOT actually organized by head; this
              is just an arbitrary 64-dim slice of d_model. Kept for compatibility
              with your existing files. Consider hook_q / hook_k / hook_v instead.)
      output: hook_z output for this head           [1, seq_len, 64]

    Returns lists of per-prompt CPU tensors.
    """
    inputs_list  = []
    outputs_list = []

    resid_name = f"blocks.{layer}.hook_resid_pre"   # [1, seq_len, 768]
    z_name     = f"blocks.{layer}.attn.hook_z"      # [1, seq_len, 12, 64]
    head_start = head * d_head
    head_end   = head_start + d_head

    for prompt in tqdm(prompts, desc=desc):
        # Use a fresh dict per prompt; capture only the two tensors we need.
        captured = {}

        def cache_resid(tensor, hook, _captured=captured):
            _captured["resid"] = tensor[:, :, head_start:head_end].detach().to("cpu")

        def cache_z(tensor, hook, _captured=captured):
            _captured["z"] = tensor[:, :, head, :].detach().to("cpu")

        # run_with_hooks auto-resets all hooks at start AND end of every call,
        # so no hook accumulation across iterations.
        with torch.no_grad():
            model.run_with_hooks(
                [prompt],
                fwd_hooks=[
                    (resid_name, cache_resid),
                    (z_name,     cache_z),
                ],
            )

        inputs_list.append(captured["resid"])   # [1, seq_len, 64]  on CPU
        outputs_list.append(captured["z"])      # [1, seq_len, 64]  on CPU

        # Help the allocator: tensors are on CPU now, but pytorch keeps a
        # reservation. Empty the cache occasionally rather than every step.

    return inputs_list, outputs_list


def pad_and_stack(tensor_list):
    """Pad variable-length sequences to the longest in the batch, then stack."""
    max_len = max(t.shape[1] for t in tensor_list)
    padded  = []
    for t in tensor_list:
        pad_len = max_len - t.shape[1]
        padded.append(torch.nn.functional.pad(t, (0, 0, 0, pad_len)))
    return torch.cat(padded, dim=0)  # [numb, max_seq_len, 64]


def save_dataset(inputs_list, outputs_list, prompts, filepath):
    inputs  = pad_and_stack(inputs_list)
    outputs = pad_and_stack(outputs_list)
    data = {
        "inputs":  inputs,   # [numb, seq_len, 64]
        "outputs": outputs,  # [numb, seq_len, 64]
        "prompts": prompts,  # list of raw strings
    }
    torch.save(data, filepath)
    print(f"  Saved → {filepath}  |  inputs: {inputs.shape}  outputs: {outputs.shape}")


def run_and_save(prompts, head, fname, desc):
    """Run, save, then aggressively free everything before the next pass."""
    inp, out = collect_activations(model, prompts, *head, desc=desc)
    save_dataset(inp, out, prompts, fname)
    del inp, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── RUN ALL 4 COMBINATIONS ────────────────────────────────────────────────────

print("\n── Collecting: IOI head  ×  IOI prompts ──")
run_and_save(ioi_prompts, IOI_HEAD,  "activations_IOI_head_IOI_promptsBIG.pt",  "IOI head | IOI prompts")

print("\n── Collecting: IOI head  ×  ABC prompts ──")
run_and_save(abc_prompts, IOI_HEAD,  "activations_IOI_head_ABC_promptsBIG.pt",  "IOI head | ABC prompts")

print("\n── Collecting: CTRL head  ×  IOI prompts ──")
run_and_save(ioi_prompts, CTRL_HEAD, "activations_CTRL_head_IOI_promptsBIG.pt", "CTRL head | IOI prompts")

print("\n── Collecting: CTRL head  ×  ABC prompts ──")
run_and_save(abc_prompts, CTRL_HEAD, "activations_CTRL_head_ABC_promptsBIG.pt", "CTRL head | ABC prompts")

# ── SANITY CHECK: confirm CTRL head is actually active on ABC ─────────────────

print("\n── Sanity check: mean output norm per head × dataset ──")
for label, fname in [
    ("IOI head  | IOI prompts", "activations_IOI_head_IOI_promptsBIG.pt"),
    ("IOI head  | ABC prompts", "activations_IOI_head_ABC_promptsBIG.pt"),
    ("CTRL head | IOI prompts", "activations_CTRL_head_IOI_promptsBIG.pt"),
    ("CTRL head | ABC prompts", "activations_CTRL_head_ABC_promptsBIG.pt"),
]:
    d    = torch.load(fname)
    norm = d["outputs"].norm(dim=-1).mean().item()
    print(f"  {label}  →  mean output norm: {norm:.4f}")

print("\nDone.")