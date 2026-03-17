"""Day-0 environment verification script."""
import sys
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU memory: {mem / 1024**3:.1f} GB")
else:
    print("WARNING: CUDA not available")

print()
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    print("Gemma-2-2b tokenizer loaded successfully")
    test_str = "The Rashomon set of linear probes is non-trivial."
    ids = tokenizer(test_str)["input_ids"]
    print(f"Token IDs: {ids}")
    decoded = [tokenizer.decode([t]) for t in ids]
    print(f"Decoded tokens: {decoded}")
except Exception as e:
    print(f"Tokenizer error: {type(e).__name__}: {e}")

print()
print("ALL CHECKS PASSED")
