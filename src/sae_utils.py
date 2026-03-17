"""Gemma Scope SAE loading, encoding, and verification utilities.

Gemma Scope uses JumpReLU SAEs:
  Encode: pre_acts = x @ W_enc + b_enc; z = ReLU(pre_acts) * (pre_acts > threshold)
  Decode: x_hat = z @ W_dec + b_dec

The b_dec subtraction is folded into b_enc during training, so encoding
applies directly to raw activations without pre-subtracting b_dec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        return mask * F.relu(pre_acts)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def load_gemma_scope_sae(layer=10, width="16k", l0=77):
    """Download and load a Gemma Scope SAE from HuggingFace.

    Returns a JumpReLUSAE module (on CPU, eval mode).
    """
    repo = "google/gemma-scope-2b-pt-res"
    filename = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
    print(f"Downloading SAE: {repo} / {filename} ...")
    path = hf_hub_download(repo_id=repo, filename=filename)

    params = np.load(path)
    print(f"Loaded keys: {list(params.keys())}")
    for k in params:
        print(f"  {k}: shape={params[k].shape}, dtype={params[k].dtype}")

    d_model, d_sae = params["W_enc"].shape
    print(f"\nd_model={d_model}, d_sae={d_sae}")

    sae = JumpReLUSAE(d_model, d_sae)
    state = {k: torch.from_numpy(v) for k, v in params.items()}
    sae.load_state_dict(state)
    sae.eval()
    return sae


def verify_sae(sae, sample_activations):
    """Run reconstruction test on a few activation vectors.

    Prints per-sample and mean reconstruction MSE and relative error.
    """
    with torch.no_grad():
        x = sample_activations.float()
        x_hat = sae(x)
        mse = (x - x_hat).pow(2).mean(dim=1)  # per sample
        norms = x.pow(2).mean(dim=1)
        relative = mse / norms

    print("\nSAE Reconstruction Verification:")
    for i in range(len(x)):
        print(f"  Sample {i}: MSE={mse[i].item():.4f}, "
              f"relative={relative[i].item():.4f} ({relative[i].item():.1%})")
    print(f"  Mean MSE: {mse.mean().item():.4f}")
    print(f"  Mean relative error: {relative.mean().item():.4f} ({relative.mean().item():.1%})")


def encode_activations(sae, activations, batch_size=256):
    """Encode a tensor of raw activations into SAE feature space.

    Returns sparse feature activations and sparsity statistics.
    """
    all_z = []
    n = len(activations)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = activations[i : i + batch_size].float()
            z = sae.encode(batch)
            all_z.append(z)
    z_all = torch.cat(all_z, dim=0)

    # Sparsity stats
    nonzero_per_example = (z_all > 0).float().sum(dim=1)
    d_sae = z_all.shape[1]
    frac_active = nonzero_per_example / d_sae

    print(f"\nSparsity statistics ({z_all.shape}):")
    print(f"  Active features per example: "
          f"mean={nonzero_per_example.mean().item():.1f}, "
          f"std={nonzero_per_example.std().item():.1f}, "
          f"min={nonzero_per_example.min().item():.0f}, "
          f"max={nonzero_per_example.max().item():.0f}")
    print(f"  Fraction active: "
          f"mean={frac_active.mean().item():.4f} ({frac_active.mean().item():.2%}), "
          f"=> {1 - frac_active.mean().item():.2%} sparse")

    return z_all
