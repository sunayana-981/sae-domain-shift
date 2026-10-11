"""
LoRA loading for DINOv2 and SigLIP2, mirroring train_sae_lora_clip.py's
load_lora_weights() for CLIP.

The adapter checkpoints here (unified_weights/{dino_dinov2-base,
siglip2-base-patch16-224}/<dataset>/16shots/seed1/lora_adapter_weights.pt)
were produced by ../unified_finetune.py, whose {'adapter':..., 'head':...,
'metadata':...} format and injection logic (LinearLoRA wrapping specific
q/k/v/o attention projections) are reproduced here directly rather than
imported, since unified_finetune.py lives outside this package and pulls in
a large CLIP_LoRA/dataset-loading dependency chain at module import time
that has nothing to do with loading a saved adapter.

ALIGN is deliberately not included: unified_finetune.py's own
_set_align_trainability documents that ALIGN's vision tower (EfficientNet)
is kept fully frozen and only the text transformer is LoRA-adapted (confirmed
directly in the saved checkpoints: every adapter key is under
text_model.*). Since this pipeline hooks vision-side activations, loading
that adapter would be a well-defined no-op for what we measure -- so ALIGN's
"LoRA-adapted" masked-FT here would be identical to its already-computed
base-backbone one; it is not the same story as DINO/SigLIP2 not being
loadable at all.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_POS_12 = {
    'top1':        [11],
    'top3':        [9, 10, 11],
    'bottom':      [0, 1, 2, 3],
    'mid':         [4, 5, 6, 7],
    'up':          [8, 9, 10, 11],
    'half-up':     [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all':         list(range(12)),
}


class LinearLoRA(nn.Module):
    """LoRA adapter wrapping any nn.Linear layer. Matches
    unified_finetune.py's LinearLoRA exactly (weight/bias frozen copies +
    lora_A/lora_B), so a checkpoint's 'adapter' state dict loads directly
    into modules built by this class with strict=True on the lora_* keys."""

    def __init__(self, existing_linear: nn.Linear, r: int = 4, lora_alpha: float = 1.0,
                 dropout_rate: float = 0.0):
        super().__init__()
        d_out, d_in = existing_linear.weight.shape
        self.scaling = lora_alpha / r
        dev = existing_linear.weight.device
        self.weight = nn.Parameter(existing_linear.weight.data.clone(), requires_grad=False)
        self.bias = (nn.Parameter(existing_linear.bias.data.clone(), requires_grad=False)
                     if existing_linear.bias is not None else None)
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_h = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora_h, self.lora_B)
        return base + self.scaling * lora


def _wrap_dino(model, r, alpha, position, params):
    """model: a Dinov2Model (or .vision_model / equivalent already unwrapped
    to expose .encoder.layer)."""
    indices = _POS_12[position]
    for i, layer in enumerate(model.encoder.layer):
        if i not in indices:
            continue
        attn_self = layer.attention.attention
        if 'q' in params:
            attn_self.query = LinearLoRA(attn_self.query, r=r, lora_alpha=alpha)
        if 'k' in params:
            attn_self.key = LinearLoRA(attn_self.key, r=r, lora_alpha=alpha)
        if 'v' in params:
            attn_self.value = LinearLoRA(attn_self.value, r=r, lora_alpha=alpha)
        if 'o' in params:
            layer.attention.output.dense = LinearLoRA(layer.attention.output.dense, r=r, lora_alpha=alpha)


def _wrap_siglip(model, r, alpha, position, params, encoder):
    """model: a SiglipModel (has .text_model and .vision_model)."""
    indices = _POS_12[position]
    branches = []
    if encoder in ('text', 'both'):
        branches.append(model.text_model.encoder.layers)
    if encoder in ('vision', 'both'):
        branches.append(model.vision_model.encoder.layers)
    projection_names = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'out_proj'}
    for layers in branches:
        for i, layer in enumerate(layers):
            if i not in indices:
                continue
            attn = layer.self_attn
            for requested, attr in projection_names.items():
                if requested not in params:
                    continue
                setattr(attn, attr, LinearLoRA(getattr(attn, attr), r=r, lora_alpha=alpha))


def load_lora_weights_multiarch(vit, lora_checkpoint_path: str, arch: str, device: str = "cuda"):
    """Loads a unified_finetune.py-format LoRA checkpoint into a
    HookedVisionTransformer wrapping a DINOv2Model or SiglipModel, mutating
    vit.model in place (module wrapping) then loading the saved lora_A/lora_B
    values. Returns vit.

    Raises ValueError for arch='align' with an explanation, since that
    checkpoint format never touches vision-side weights in this codebase
    (see module docstring) -- callers should treat ALIGN as base-backbone
    only rather than call this function for it.
    """
    if arch == "align":
        raise ValueError(
            "ALIGN's LoRA adapters (unified_finetune.py) only touch the text "
            "transformer -- the vision tower stays frozen by design (see "
            "_set_align_trainability there). Loading it would not change the "
            "vision-side activations this pipeline hooks, so it is "
            "intentionally not wired up here; use the base backbone instead."
        )
    if arch not in ("dino", "siglip2"):
        raise ValueError(f"load_lora_weights_multiarch does not support arch={arch!r}")

    ckpt_path = Path(lora_checkpoint_path)
    assert ckpt_path.exists(), f"[FATAL] LoRA checkpoint not found: {ckpt_path}"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not (isinstance(ckpt, dict) and "adapter" in ckpt and "metadata" in ckpt):
        raise ValueError(f"Expected a unified_finetune.py-format checkpoint (adapter+metadata keys), got: "
                          f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    meta = ckpt["metadata"]
    r, alpha = meta["r"], meta["alpha"]
    position = meta.get("position", "all")
    params = meta.get("params", ["q", "k", "v"])
    print(f"[DEBUG] multiarch LoRA metadata: {meta}")

    model = vit.model  # the raw HF model wrapped by HookedVisionTransformer
    if arch == "dino":
        _wrap_dino(model, r, alpha, position, params)
    else:
        _wrap_siglip(model, r, alpha, position, params, meta.get("encoder", "both"))
    model.to(device)

    missing, unexpected = model.load_state_dict(ckpt["adapter"], strict=False)
    # Only lora_A/lora_B are in the checkpoint; every other parameter name
    # (backbone weights, wrapped-linear's own frozen weight/bias copies) is
    # expected to show up as "missing" here since load_state_dict is checking
    # against the full model, not just the adapter. What must NOT happen is
    # any adapter key itself being unmatched.
    unmatched_adapter_keys = [k for k in ckpt["adapter"].keys() if k in missing]
    if unmatched_adapter_keys:
        raise RuntimeError(f"LoRA adapter keys failed to load (module wrapping did not match "
                            f"checkpoint's module paths): {unmatched_adapter_keys[:5]}...")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading LoRA adapter: {unexpected[:5]}...")
    print(f"[DEBUG] Loaded {len(ckpt['adapter'])} LoRA tensors into {arch} model "
          f"(position={position}, params={params}).")
    return vit
