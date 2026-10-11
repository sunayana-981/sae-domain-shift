"""
Building blocks for joint LoRA + adapter-SAE training.

The frozen G-SAE keeps every concept it already has. A separate *adapter SAE*
adds latents whose decoder directions are confined to
the subspace spanned by the LoRA shift (adapted − base activations at the SAE
layer). It reads the same input as the G-SAE; this constraint alone does not
make its latents identify the shift rather than target content. Together,

    x̂ = G-SAE(x) + adapter_decode(adapter_encode(x)),

which is exactly one standard ReLU SAE with d_sae = d_G + d_adapter whose first
d_G units are frozen. `merge_into_sae` exports it in that form, with a
`protected_mask`, so every existing loader and evaluation script works on it.

Contents:
  LoRALinear / apply_vision_lora / set_lora_enabled / export_lora
      LoRA on HF CLIP vision attention q/k/v, saved in the CLIP_LoRA format
      that tasks.train_sae_lora_clip.load_lora_weights merges.
  shift_subspace
      Orthonormal basis of the LoRA shift from paired activations.
  FrozenReLUSAE
      The G-SAE as fixed tensors (encode/decode with gradients to the input).
  AdapterSAE
      Subspace-constrained ReLU SAE (W_dec = C Uᵀ), with refresh and re-seed.
  merge_into_sae
      G-SAE ⊕ adapter → checkpoint dict for SparseAutoencoder.
  scale_grad
      x.detach() + γ (x − x.detach()): forward value x, gradient scaled by γ.
"""

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# LoRA on HF CLIP vision attention
# ═══════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """y = base(x) + (α/√r) · B A dropout(x). B starts at zero, so the wrapped
    layer is initially identical to the base layer. The α/√r scaling matches
    the CLIP_LoRA checkpoint loader."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.25):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / math.sqrt(r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return y


_PROJ = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj'}


def apply_vision_lora(clip_model, r: int = 2, alpha: float = 1, dropout: float = 0.25,
                      params: Sequence[str] = ('q', 'k', 'v'),
                      max_layer: Optional[int] = None) -> List[Tuple[int, str, LoRALinear]]:
    """Freeze the whole CLIP model and wrap q/k/v of vision layers 0..max_layer
    (all layers if None). Layers always start at 0, so the exported keys stay
    consecutive, as the CLIP_LoRA loader requires. Returns
    [(layer_index, proj_name, module)] in layer order."""
    for p in clip_model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for i, layer in enumerate(clip_model.vision_model.encoder.layers):
        if max_layer is not None and i > max_layer:
            break
        for p in params:
            name = _PROJ[p]
            mod = LoRALinear(getattr(layer.self_attn, name), r, alpha, dropout)
            setattr(layer.self_attn, name, mod)
            wrapped.append((i, name, mod))
    return wrapped


def set_lora_enabled(wrapped, enabled: bool):
    for _, _, mod in wrapped:
        mod.enabled = enabled


def export_lora(wrapped, r: int, alpha: float, params: Sequence[str] = ('q', 'k', 'v')) -> dict:
    """CLIP_LoRA checkpoint format (`weights` + `metadata`, encoder='vision').
    Layer keys are consecutive, which the loader relies on."""
    weights: Dict[str, dict] = {}
    for i, name, mod in wrapped:
        weights.setdefault(f'layer_{i}', {})[name] = {
            'w_lora_A': mod.lora_A.detach().cpu().clone(),
            'w_lora_B': mod.lora_B.detach().cpu().clone()}
    return {'weights': weights,
            'metadata': {'r': r, 'alpha': alpha, 'encoder': 'vision', 'params': list(params), 'position': 'all'}}


# ═══════════════════════════════════════════════════════════════════════════
# Shift subspace, frozen G-SAE, adapter SAE
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def shift_subspace(delta: torch.Tensor, energy: float = 0.95, k_max: int = 128,
                   k: Optional[int] = None) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """SVD of the shift tokens Δ (n × d). Returns (U d×k, k, singular values).
    k is the smallest number of components holding `energy` of Σs², capped at
    k_max, unless k is given (used on refresh to keep the size fixed)."""
    _, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    if k is None:
        frac = (s ** 2).cumsum(0) / (s ** 2).sum().clamp(min=1e-30)
        k = int(min(k_max, int((frac < energy).sum().item()) + 1))
    return vh[:k].T.contiguous(), k, s


class FrozenReLUSAE(nn.Module):
    """A standard ReLU SAE held as fixed buffers. Gradients flow to the input
    (for joint training) but never to these weights."""

    def __init__(self, state_dict: dict, device):
        super().__init__()
        for k in ('W_enc', 'b_enc', 'W_dec', 'b_dec'):
            self.register_buffer(k, state_dict[k].detach().to(device).float().clone())

    @property
    def d_sae(self):
        return self.W_enc.shape[1]

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec


class AdapterSAE(nn.Module):
    """ReLU SAE whose decoder rows live in span(U): W_dec = C Uᵀ, rows of C
    unit-norm (so decoder rows are unit-norm because U is orthonormal). Reads
    x − b_dec of the G-SAE; adds no bias of its own on decode."""

    def __init__(self, U: torch.Tensor, coords: torch.Tensor, b_dec: torch.Tensor, enc_scale: float = 1.0):
        super().__init__()
        self.register_buffer('U', U.detach().clone().float())                  # d_in × k
        self.register_buffer('b_dec', b_dec.detach().clone().float())          # shared with G-SAE
        c = F.normalize(coords.detach().float(), dim=1)
        self.C = nn.Parameter(c.clone())                                        # d_sae × k
        self.W_enc = nn.Parameter((c @ self.U.T).T.clone() * enc_scale)         # d_in × d_sae
        self.b_enc = nn.Parameter(torch.zeros(c.shape[0], device=c.device))

    @property
    def d_sae(self):
        return self.C.shape[0]

    @property
    def W_dec(self):
        return self.C @ self.U.T                                                # d_sae × d_in

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec

    @torch.no_grad()
    def normalize_(self):
        self.C.data = F.normalize(self.C.data, dim=1)

    @torch.no_grad()
    def refresh_subspace_(self, U_new: torch.Tensor):
        """Move to a new basis of the same size, keeping each latent's decoder
        direction as close as the new subspace allows."""
        assert U_new.shape == self.U.shape, 'refresh keeps the subspace size fixed'
        self.C.data = F.normalize(self.C.data @ (self.U.T @ U_new.float()), dim=1)
        self.U.copy_(U_new.float())

    @torch.no_grad()
    def reseed_(self, idx: torch.Tensor, coords: torch.Tensor, enc_scale: float = 0.2):
        """Re-initialise latents `idx` to the given subspace coordinates."""
        if idx.numel() == 0:
            return
        c = F.normalize(coords.float(), dim=1)
        self.C.data[idx] = c
        self.W_enc.data[:, idx] = (c @ self.U.T).T * enc_scale
        self.b_enc.data[idx] = 0.0


def scale_grad(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Same forward value as x; backward gradient multiplied by gamma."""
    if gamma == 1.0:
        return x
    d = x.detach()
    return d if gamma == 0.0 else d + gamma * (x - d)


@torch.no_grad()
def calibration_scale(residual: torch.Tensor, decoded: torch.Tensor, max_scale: float = 0.0):
    """Nonnegative least-squares gain; an optional cap prevents enormous gains
    when newly seeded latents barely activate on calibration tokens."""
    if max_scale < 0:
        raise ValueError('max_scale must be nonnegative')
    raw = ((residual * decoded).sum() / decoded.square().sum().clamp(min=1e-12)).clamp(min=0)
    return (raw.clamp(max=max_scale) if max_scale > 0 else raw), raw


@torch.no_grad()
def youden_thresholds(target: torch.Tensor, reference: torch.Tensor,
                      min_target_fire: float = 0.05, init_fire_rate: float = 0.05):
    """Choose thresholds for strict `activation > threshold`, evaluating whole
    tie blocks. Return the separation actually achieved by those thresholds.
    Inputs are image-max preactivations, [images, latents]."""
    if not 0 < min_target_fire <= 1 or not 0 < init_fire_rate <= 1:
        raise ValueError('fire rates must be in (0, 1]')
    if target.ndim != 2 or reference.ndim != 2 or target.shape[1] != reference.shape[1] or len(target) == 0:
        raise ValueError('expected nonempty target and matching [images, latents] tensors')
    if not torch.isfinite(target).all() or not torch.isfinite(reference).all():
        raise ValueError('threshold inputs must be finite')
    if len(reference) == 0:
        kth = max(1, int(round((1 - init_fire_rate) * len(target))))
        theta = target.kthvalue(kth, dim=0).values
        return theta, torch.zeros_like(theta)
    values = torch.cat([target, reference])
    labels = torch.cat([target.new_ones(len(target)), target.new_zeros(len(reference))])
    ordered, indices = values.sort(dim=0, descending=True)
    positive = labels[indices]
    tpr = positive.cumsum(0) / len(target)
    fpr = (1 - positive).cumsum(0) / len(reference)
    ends = torch.ones_like(ordered, dtype=torch.bool)
    ends[:-1] = ordered[:-1] > ordered[1:]
    score = (tpr - fpr).masked_fill(~ends | (tpr < min_target_fire), -torch.inf)
    best = score.argmax(0)
    current = ordered.gather(0, best[None])[0]
    # The representable value directly below the tie block gives exactly the
    # prefix evaluated above, including when adjacent scores differ by one ULP.
    theta = torch.nextafter(current, torch.full_like(current, -torch.inf))
    return theta, score.gather(0, best[None])[0]


def backward_projected_aux(task_loss, aux_loss, lora_params, adapter_params, max_ratio=0.25):
    """Backpropagate task/retention normally; project conflicting SAE gradients
    off that gradient and cap their global norm. Adapter gradients are unchanged.
    This constrains raw gradients; Adam's preconditioning can still rotate steps."""
    if max_ratio < 0:
        raise ValueError('max_ratio must be nonnegative')
    params = list(lora_params) + list(adapter_params)
    auxiliary = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
    task_loss.backward()
    task = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in lora_params]
    aux = [torch.zeros_like(p) if g is None else g for p, g in zip(lora_params, auxiliary)]
    task_sq = sum(g.float().square().sum() for g in task)
    aux_sq = sum(g.float().square().sum() for g in aux)
    dot = sum((a.float() * b.float()).sum() for a, b in zip(task, aux))
    coefficient = dot.clamp(max=0) / task_sq.clamp(min=1e-30)
    projected = [b - coefficient * a for a, b in zip(task, aux)]
    projected_norm = sum(g.float().square().sum() for g in projected).sqrt()
    scale = (max_ratio * task_sq.sqrt() / projected_norm.clamp(min=1e-30)).clamp(max=1)
    for p, a, b in zip(lora_params, task, projected):
        p.grad = a + scale * b
    for p, g in zip(adapter_params, auxiliary[len(lora_params):]):
        if g is not None:
            p.grad = g if p.grad is None else p.grad + g
    return dict(sae_task_cosine=float(dot / (task_sq * aux_sq).sqrt().clamp(min=1e-30)),
                sae_grad_scale=float(scale), task_grad_norm=float(task_sq.sqrt()),
                sae_grad_norm=float(aux_sq.sqrt()))


def reset_adam_rows_(optimizer: torch.optim.Optimizer, param: nn.Parameter, idx: torch.Tensor, dim: int):
    """Zero Adam moments for the slices of `param` at `idx` along `dim`."""
    st = optimizer.state.get(param, {})
    for key in ('exp_avg', 'exp_avg_sq'):
        if key in st:
            st[key].index_fill_(dim, idx.to(st[key].device), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Export: G-SAE ⊕ adapter as one SparseAutoencoder checkpoint
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def merge_into_sae(gsae_ckpt: dict, adapter: AdapterSAE, extra: Optional[dict] = None) -> dict:
    """Checkpoint dict loadable by eval_masked_sae.load_masked_sae /
    tasks.utils.load_sae: cfg (d_sae = d_G + d_adapter), state_dict and a
    protected_mask marking the frozen G-SAE units."""
    cfg = gsae_ckpt.get('cfg', gsae_ckpt.get('config'))
    cfg = copy.copy(cfg if isinstance(cfg, dict) else dict(vars(cfg)))
    sd = gsae_ckpt['state_dict']
    if cfg.get('gated_sae') or cfg.get('topk_sae') or cfg.get('jumprelu_sae'):
        raise NotImplementedError('merge_into_sae supports the standard ReLU G-SAE only')
    d_g = sd['W_enc'].shape[1]
    W_enc = torch.cat([sd['W_enc'].float().cpu(), adapter.W_enc.detach().float().cpu()], dim=1)
    b_enc = torch.cat([sd['b_enc'].float().cpu(), adapter.b_enc.detach().float().cpu()])
    W_dec = torch.cat([sd['W_dec'].float().cpu(), adapter.W_dec.detach().float().cpu()], dim=0)
    new_sd = dict(sd)
    new_sd.update(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=sd['b_dec'].float().cpu())
    cfg['d_sae'] = int(W_enc.shape[1])
    mask = torch.zeros(cfg['d_sae'], dtype=torch.bool)
    mask[:d_g] = True
    out = {'cfg': cfg, 'state_dict': new_sd, 'protected_mask': mask}
    if extra:
        out.update(extra)
    return out
