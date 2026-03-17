"""Data loading and activation extraction for BeaverTails safety classification."""

import time
import torch
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_balanced_beavertails(n_per_class=1000, seed=42):
    """Load BeaverTails and return a balanced subset with train/test split.

    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    print("Loading BeaverTails dataset (330k_train split) ...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

    safe_idx = [i for i, ex in enumerate(ds) if ex["is_safe"]]
    unsafe_idx = [i for i, ex in enumerate(ds) if not ex["is_safe"]]
    print(f"Full dataset: {len(safe_idx)} safe, {len(unsafe_idx)} unsafe")

    rng = np.random.RandomState(seed)
    safe_sample = rng.choice(safe_idx, size=min(n_per_class, len(safe_idx)), replace=False)
    unsafe_sample = rng.choice(unsafe_idx, size=min(n_per_class, len(unsafe_idx)), replace=False)

    indices = np.concatenate([safe_sample, unsafe_sample])
    texts = [ds[int(i)]["response"] for i in indices]
    labels = [1] * len(safe_sample) + [0] * len(unsafe_sample)
    print(f"Sampled {len(safe_sample)} safe + {len(unsafe_sample)} unsafe = {len(texts)} total")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels,
    )
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    return train_texts, test_texts, train_labels, test_labels


def extract_activations(texts, model, tokenizer, layer_idx=10, batch_size=8):
    """Extract mean-pooled residual stream activations at a given layer.

    HuggingFace hidden_states tuple has length (num_layers + 1).
    Index 0 = embedding output; index k = output after transformer block k.

    Returns:
        Tensor of shape [len(texts), hidden_dim]
    """
    all_vecs = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    t0 = time.time()

    for b in range(n_batches):
        batch_texts = texts[b * batch_size : (b + 1) * batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]  # [B, seq_len, hidden_dim]
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [B, hidden_dim]
        all_vecs.append(pooled.cpu().float())

        if (b + 1) % 25 == 0 or (b + 1) == n_batches:
            elapsed = time.time() - t0
            print(f"  Batch {b + 1}/{n_batches}  ({elapsed:.1f}s)")

    result = torch.cat(all_vecs, dim=0)
    print(f"Extraction complete: {result.shape} in {time.time() - t0:.1f}s")
    return result


def load_model_and_tokenizer(model_name="google/gemma-2-2b"):
    """Load model and tokenizer for activation extraction."""
    print(f"Loading {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded. GPU memory allocated: {mem:.2f} GB")
    return model, tokenizer
