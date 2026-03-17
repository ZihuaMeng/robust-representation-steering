"""Verify Gemma-2 2B loading and residual stream activation extraction.

HuggingFace models return hidden_states as a tuple of length (num_layers + 1).
Index 0 is the embedding layer output; index k (for k >= 1) is the output of
the k-th transformer block. So hidden_states[10] gives activations after
transformer block 10.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_LAYERS = [10, 15, 20]
EXPECTED_HIDDEN_DIM = 2304

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading google/gemma-2-2b ...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
except torch.cuda.OutOfMemoryError:
    print("OOM: Model too large for available GPU memory.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

mem_alloc = torch.cuda.memory_allocated() / 1024**3
mem_resv = torch.cuda.memory_reserved() / 1024**3
print(f"Model loaded. GPU memory — allocated: {mem_alloc:.2f} GB, reserved: {mem_resv:.2f} GB")

# ── Model config validation ──────────────────────────────────────────────────
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Config: num_hidden_layers={num_layers}, hidden_size={hidden_size}")

all_ok = True

if hidden_size != EXPECTED_HIDDEN_DIM:
    print(f"WARNING: expected hidden_size={EXPECTED_HIDDEN_DIM}, got {hidden_size}")
    all_ok = False

for layer_idx in TARGET_LAYERS:
    if layer_idx > num_layers:
        print(f"WARNING: target layer {layer_idx} exceeds num_hidden_layers={num_layers}")
        all_ok = False

# ── Tokenize test prompts ────────────────────────────────────────────────────
prompts = [
    "The concept of safety in language models",
    "Linear probes detect features in residual streams",
    "Adversarial weight perturbation explores model multiplicity",
]

inputs = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
).to(model.device)

print(f"Input shape: {inputs['input_ids'].shape}")

# ── Forward pass ─────────────────────────────────────────────────────────────
print("Running forward pass ...")
try:
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
except torch.cuda.OutOfMemoryError:
    print("OOM during forward pass.")
    sys.exit(1)

hidden_states = outputs.hidden_states  # length = num_layers + 1
print(f"Hidden states tuple length: {len(hidden_states)} (num_layers + 1 = {num_layers + 1})")

# ── Extract and validate target layers ───────────────────────────────────────
print("\n" + "=" * 60)
for layer_idx in TARGET_LAYERS:
    h = hidden_states[layer_idx].float()  # cast for stats
    batch, seq_len, hdim = h.shape
    print(f"Layer {layer_idx:>2d} | shape: [{batch}, {seq_len}, {hdim}]")
    print(f"           mean={h.mean().item():.6f}  std={h.std().item():.6f}")
    print(f"           min={h.min().item():.6f}   max={h.max().item():.6f}")
    if hdim != EXPECTED_HIDDEN_DIM:
        print(f"  !! hidden_dim mismatch: expected {EXPECTED_HIDDEN_DIM}, got {hdim}")
        all_ok = False
    if torch.isnan(h).any():
        print(f"  !! NaN detected in layer {layer_idx} activations")
        all_ok = False
print("=" * 60)

# ── Final memory report ─────────────────────────────────────────────────────
mem_alloc = torch.cuda.memory_allocated() / 1024**3
mem_resv = torch.cuda.memory_reserved() / 1024**3
print(f"\nPost-forward GPU memory — allocated: {mem_alloc:.2f} GB, reserved: {mem_resv:.2f} GB")

if all_ok:
    print("\nACTIVATION PIPELINE VERIFIED")
else:
    print("\nSome checks failed — see warnings above.")
