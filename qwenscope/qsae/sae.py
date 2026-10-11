"""Qwen-Scope Top-K SAE: checkpoint loading + a trainable nn.Module.

Qwen-Scope checkpoints (Qwen/SAE-Res-Qwen3.5-*) are plain state dicts with
four tensors per layer file (`layer{L}.sae.pt`):

    W_enc  (d_sae, d_model)
    b_enc  (d_sae,)
    W_dec  (d_model, d_sae)
    b_dec  (d_model,)

and a repo-level `config.json` giving `model_type: topk_sae`, `d_model`,
`d_sae`, `k` and `hook_point: resid_post`.

The card does not pin down whether the encoder pre-activation is computed on
`x` or on the centred `x - b_dec`, so `encode_variant` selects between the two
and `scripts/probe_sae_convention.py` measures both. Measured on Qwen3.5-2B-Base
layer 20 over 16k fineweb tokens, `raw` wins clearly -- FVE 0.772 / cosine 0.893
versus 0.572 / 0.798 for `centered` -- and L0 comes out at exactly k=50, so
`raw` is the default. The probe also confirms the released decoder columns are
exactly unit-norm, which is why training renormalises them after every step.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

ENCODE_VARIANTS = ("centered", "raw")


@dataclass
class SAEConfig:
    d_model: int
    d_sae: int
    k: int
    layer: int
    hook_point: str = "resid_post"
    encode_variant: str = "raw"
    source: str = ""

    @property
    def expansion_factor(self) -> int:
        return self.d_sae // self.d_model

    def to_dict(self):
        return dict(
            d_model=self.d_model, d_sae=self.d_sae, k=self.k, layer=self.layer,
            hook_point=self.hook_point, encode_variant=self.encode_variant,
            expansion_factor=self.expansion_factor, source=self.source,
        )


class TopKSAE(nn.Module):
    """Top-K sparse autoencoder with Qwen-Scope's parameter layout.

    forward(x) -> (x_hat, feature_acts, pre_acts)

    `feature_acts` is the k-sparse code (zeros everywhere outside the top-k),
    so L0 == k by construction on every token; `pre_acts` is the dense
    pre-activation, which the auxiliary dead-latent loss needs.
    """

    def __init__(self, cfg: SAEConfig, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.d_sae = cfg.d_sae
        self.k = cfg.k
        self.encode_variant = cfg.encode_variant

        self.W_enc = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model, dtype=dtype))

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.to(device)
        self.device = device

    # ---------------------------------------------------------------- encode
    def pre_acts(self, x):
        h = x - self.b_dec if self.encode_variant == "centered" else x
        return F.linear(h, self.W_enc, self.b_enc)

    def topk_mask(self, pre):
        """Indices/values of the k largest pre-activations, ReLU-clipped.

        Selection is global across all d_sae units for a given token -- which
        is exactly why masked fine-tuning of a Top-K SAE cannot use a
        split-forward over free/protected slices (see masked_trainer).
        """
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return vals.relu(), idx

    def encode(self, x):
        pre = self.pre_acts(x)
        vals, idx = self.topk_mask(pre)
        acts = torch.zeros_like(pre).scatter_(-1, idx, vals)
        return acts, pre

    def decode(self, acts):
        return F.linear(acts, self.W_dec, self.b_dec)

    def forward(self, x):
        acts, pre = self.encode(x)
        return self.decode(acts), acts, pre

    # ------------------------------------------------------------ maintenance
    @torch.no_grad()
    def normalize_decoder(self):
        """Rescale decoder columns (one per unit) back to unit norm."""
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def decoder_norms(self):
        return self.W_dec.data.norm(dim=0)

    def state_dict_qwenscope(self):
        """Save in the same four-tensor layout Qwen-Scope ships."""
        return {
            "W_enc": self.W_enc.detach().cpu(),
            "b_enc": self.b_enc.detach().cpu(),
            "W_dec": self.W_dec.detach().cpu(),
            "b_dec": self.b_dec.detach().cpu(),
        }

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sae": self.state_dict_qwenscope(), "cfg": self.cfg.to_dict(),
             "extra": extra or {}},
            path,
        )
        return str(path)


# ═════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════

def _load_tensors(obj):
    """Accepts a raw Qwen-Scope state dict or one of our own save() bundles."""
    if isinstance(obj, dict) and "sae" in obj and "W_enc" in obj["sae"]:
        return obj["sae"], obj.get("cfg", {})
    if isinstance(obj, dict) and "W_enc" in obj:
        return obj, {}
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("cfg", {})
    raise ValueError(f"unrecognised SAE checkpoint keys: {list(obj)[:8]}")


def load_qwenscope_sae(repo_id, layer, device="cuda", encode_variant="raw",
                       dtype=torch.float32):
    """Download + load one layer of a Qwen-Scope release."""
    cfg_path = hf_hub_download(repo_id, "config.json")
    raw_cfg = json.load(open(cfg_path))
    if raw_cfg.get("model_type") != "topk_sae":
        raise ValueError(f"{repo_id} is {raw_cfg.get('model_type')}, expected topk_sae")
    if layer not in raw_cfg["layers"]:
        raise ValueError(f"layer {layer} not in {repo_id} (has {raw_cfg['layers']})")

    w_path = hf_hub_download(repo_id, f"layer{layer}.sae.pt")
    tensors, _ = _load_tensors(torch.load(w_path, map_location="cpu"))

    cfg = SAEConfig(
        d_model=raw_cfg["d_model"], d_sae=raw_cfg["d_sae"], k=raw_cfg["k"],
        layer=layer, hook_point=raw_cfg.get("hook_point", "resid_post"),
        encode_variant=encode_variant, source=f"{repo_id}:layer{layer}",
    )
    sae = TopKSAE(cfg, dtype=dtype, device=device)
    _copy_in(sae, tensors, dtype, device)
    return sae, raw_cfg


def load_local_sae(path, device="cuda", encode_variant=None, dtype=torch.float32):
    """Load an SAE we trained/fine-tuned ourselves."""
    obj = torch.load(path, map_location="cpu")
    tensors, saved_cfg = _load_tensors(obj)
    d_sae, d_model = tensors["W_enc"].shape
    cfg = SAEConfig(
        d_model=saved_cfg.get("d_model", d_model),
        d_sae=saved_cfg.get("d_sae", d_sae),
        k=saved_cfg.get("k", 50),
        layer=saved_cfg.get("layer", -1),
        hook_point=saved_cfg.get("hook_point", "resid_post"),
        encode_variant=encode_variant or saved_cfg.get("encode_variant", "raw"),
        source=str(path),
    )
    sae = TopKSAE(cfg, dtype=dtype, device=device)
    _copy_in(sae, tensors, dtype, device)
    return sae


def _copy_in(sae, tensors, dtype, device):
    with torch.no_grad():
        sae.W_enc.copy_(tensors["W_enc"].to(dtype))
        sae.b_enc.copy_(tensors["b_enc"].to(dtype))
        sae.W_dec.copy_(tensors["W_dec"].to(dtype))
        sae.b_dec.copy_(tensors["b_dec"].to(dtype))
    sae.to(device)
    sae.device = device
