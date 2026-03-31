"""Generate steered completions for BeaverTails prompts via residual deltas."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import load_balanced_beavertails, load_model_and_tokenizer
from steering import naive_delta, robust_delta, robust_delta_dynamic


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


class ResidualSteeringHook:
    """Forward hook that adds a delta to the final token at a transformer layer."""

    def __init__(self, model, layer_idx, delta):
        self.layer = model.model.layers[layer_idx]
        self.delta = delta
        self.handle = None

    def __enter__(self):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output
            hidden = hidden.clone()
            hidden[:, -1, :] += self.delta
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _load_data(seed, n_per_class, activations_path):
    _, test_texts, _, test_labels, _, test_meta = load_balanced_beavertails(
        n_per_class=n_per_class, seed=seed, return_metadata=True,
    )
    data = torch.load(activations_path, map_location="cpu", weights_only=False)
    assert len(test_texts) == data["test_X"].shape[0], "Test split mismatch vs activations"
    return data, test_meta, torch.tensor(test_labels)


def _generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p,
              delta=None, layer_idx=10, apply_chat_template=False):
    device = model.device
    if apply_chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        if delta is None:
            out = model.generate(**inputs, **gen_kwargs)
        else:
            delta_device = delta.to(device=device, dtype=model.dtype)
            with ResidualSteeringHook(model, layer_idx, delta_device):
                out = model.generate(**inputs, **gen_kwargs)
    gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def _compute_delta(method, weight, bias, x, h_inv, epsilon, threshold, reverse=False):
    if method == "naive":
        d, _ = naive_delta(weight, bias, x, threshold)
    elif method == "robust":
        d = robust_delta(weight, bias, x, h_inv, epsilon, threshold)
    elif method == "dynamic":
        d = robust_delta_dynamic(weight, bias, x, h_inv, epsilon, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    return -d if reverse else d


def run(args):
    sfx = "" if args.model_variant == "pt" else f"_{args.model_variant}"
    activations_path = OUTPUT_DIR / f"beavertails_activations_layer10_{args.pooling}{sfx}.pt"
    baseline_path = OUTPUT_DIR / f"baseline_probe_layer10_{args.pooling}{sfx}.pt"
    hessian_path = OUTPUT_DIR / f"hessian_layer10_{args.pooling}{sfx}.pt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rev_sfx = "_reverse" if args.reverse else ""
    out_dir = OUTPUT_DIR / f"steering_candidates{sfx}{rev_sfx}"
    out_dir.mkdir(exist_ok=True)

    print("Loading cached activations, baseline probe, and Hessian ...")
    data, test_meta, test_labels = _load_data(args.seed, args.n_per_class, activations_path)
    test_X = data["test_X"].double()
    test_y = data["test_y"].long()

    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    weight = baseline["weight"].double()
    bias = baseline["bias"].double()

    hessian = torch.load(hessian_path, map_location="cpu", weights_only=False)
    H_inv = hessian["H_inv"].double()

    print("Loading Gemma model for generation ...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    tokenizer.padding_side = "left"

    logits = (test_X @ weight) + bias
    unsafe_mask = logits < args.threshold
    mask = unsafe_mask if args.only_unsafe else torch.ones_like(unsafe_mask, dtype=torch.bool)
    indices = torch.nonzero(mask, as_tuple=True)[0]
    if len(indices) == 0:
        print("No matching examples found.")
        return

    if args.max_examples:
        indices = indices[: args.max_examples]

    torch.manual_seed(args.generation_seed)

    records = {m: [] for m in args.methods}
    start = time.time()

    for counter, idx in enumerate(indices, 1):
        i = idx.item()
        meta = test_meta[i]
        prompt = meta.get("prompt", meta.get("metadata", {}).get("prompt", ""))
        original_response = meta.get("metadata", {}).get("response", meta.get("text", ""))

        if not prompt:
            prompt = original_response  # fallback

        baseline_text = _generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            delta=None,
            layer_idx=args.layer_idx,
            apply_chat_template=args.apply_chat_template,
        )

        x = test_X[i]
        for method in args.methods:
            delta = _compute_delta(method, weight, bias, x, H_inv, args.epsilon, args.threshold,
                                   reverse=args.reverse)
            steered_text = _generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                delta=delta,
                layer_idx=args.layer_idx,
                apply_chat_template=args.apply_chat_template,
            )
            records[method].append({
                "example_id": i,
                "dataset_index": meta.get("dataset_index", i),
                "prompt": prompt,
                "original_response": original_response,
                "baseline_completion": baseline_text,
                "steered_completion": steered_text,
                "method": method,
                "delta_norm": delta.norm().item(),
                "score_before": logits[i].item(),
                "label": int(test_y[i].item()),
                "is_safe": bool(test_labels[i].item()),
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.generation_seed,
                },
            })

        if counter % 5 == 0 or counter == len(indices):
            print(f"Processed {counter}/{len(indices)} examples")

    for method, rows in records.items():
        if not rows:
            continue
        path = out_dir / f"{method}_steering.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} rows to {path}")

    elapsed = time.time() - start
    print(f"Steering inference complete in {elapsed/60:.1f} min")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate steered Gemma completions.")
    parser.add_argument("--methods", nargs="+", default=["naive", "robust"],
                        choices=["naive", "robust", "dynamic"],
                        help="Steering methods to evaluate.")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "all"],
                        help="Pooling mode matching the probe training (mean/last/all).")
    parser.add_argument("--model-name", default="google/gemma-2-2b",
                        help="HuggingFace model to decode with.")
    parser.add_argument("--model-variant", default="pt",
                        help="Variant tag for selecting artifact files (e.g. 'pt' or 'it').")
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Wrap prompts in the tokenizer's chat template (required for instruct models).")
    parser.add_argument("--reverse", action="store_true",
                        help="Negate the steering delta (push toward unsafe rather than safe).")
    parser.add_argument("--layer-idx", type=int, default=10,
                        help="Residual layer index for steering (default=10).")
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Rashomon epsilon used during robust delta computation.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Logit threshold defining unsafe examples (sigmoid=0.5).")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Number of test examples to process (post-filter).")
    parser.add_argument("--only-unsafe", dest="only_unsafe", action="store_true",
                        help="Restrict to examples below the logit threshold (default).")
    parser.add_argument("--include-safe", dest="only_unsafe", action="store_false",
                        help="Evaluate all test examples regardless of baseline logit.")
    parser.set_defaults(only_unsafe=True)
    parser.add_argument("--n-per-class", type=int, default=1000,
                        help="Number of samples per class used when caching activations.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for generation.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p nucleus sampling for generation.")
    parser.add_argument("--generation-seed", type=int, default=0,
                        help="Seed for stochastic decoding.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used for BeaverTails split (matches training).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
