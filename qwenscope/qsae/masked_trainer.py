"""Masked fine-tuning of a Top-K SAE, ported from patchsae.

Same two steps as `patchsae/src/sae_training/masked_sae_trainer.py`:

  Step A  estimate per-unit activity on a *reference* distribution, and mark
          the top `protect_frac` most-active units as protected;
  Step B  fine-tune on the target distribution with those units frozen -- no
          gradient reaches them, and their parameters are restored bit-exactly
          after every optimizer step, so decoder renormalisation cannot move
          them either.

One deliberate difference from the vision code, forced by the architecture:
patchsae's fast split-forward (run the protected slice under `no_grad`) is only
valid for ReLU SAEs. Qwen-Scope ships **Top-K** SAEs, and top-k selection is
global across all d_sae units for a token, so taking top-k within the free and
protected slices separately would select a different active set than the dense
forward -- a different algorithm, not a faster one. patchsae's own trainer
falls back to the dense-forward + gradient-zeroing path for `topk` for exactly
this reason, and that is the only path implemented here.

Because L0 == k on every token by construction, Top-K needs no L1 term; dead
units are revived with Gao et al.'s auxiliary reconstruction loss (`aux_k`)
rather than ghost gradients.
"""

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm


# ═════════════════════════════════════════════════════════════════════════
# Step A: activity estimation and unit selection
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_unit_activity(sae, activation_store, n_batches=50, verbose=True):
    """Mean |activation| per SAE unit over `n_batches` batches.

    Identical statistic to the vision version, on the reference corpus. Also
    returns the firing *rate* per unit, which is the more informative quantity
    for a Top-K SAE (where a unit is either exactly 0 or above threshold).
    """
    sae.eval()
    d_sae = sae.d_sae
    acc = torch.zeros(d_sae, device=sae.device, dtype=torch.float32)
    fired = torch.zeros(d_sae, device=sae.device, dtype=torch.float32)
    n_tokens = 0

    it = range(n_batches)
    if verbose:
        it = tqdm(it, desc="activity", leave=False)
    for _ in it:
        x = activation_store.get_batch_activations()
        _, acts, _ = sae(x)
        flat = acts.reshape(-1, d_sae)
        acc += flat.abs().mean(dim=0)
        fired += (flat > 0).float().sum(dim=0)
        n_tokens += flat.shape[0]

    return acc / n_batches, fired / max(n_tokens, 1)


def compute_protected_mask(sae, activation_store, protect_frac=0.2, n_batches=50,
                           device="cuda", protect_mode="top", seed=42):
    """Freeze `protect_frac` of the units; return a bool mask [d_sae].

    `protect_mode` holds the free-capacity budget fixed while varying *which*
    units get it, so "how much capacity adaptation gets" is separable from
    "whether it gets the unimportant units":

        top     freeze the most-active units  -> free set is the bottom tail
                (the method: preserve reference-domain concepts)
        bottom  freeze the least-active units -> free set is the top units
        random  freeze a uniformly random subset
    """
    if protect_mode not in ("top", "bottom", "random"):
        raise ValueError(f"protect_mode must be top/bottom/random, got {protect_mode!r}")
    if not 0.0 <= protect_frac <= 1.0:
        raise ValueError(f"protect_frac must be in [0, 1], got {protect_frac}")

    print(f"[MaskedTrainer] Estimating unit activity over {n_batches} batches...")
    activity, fire_rate = estimate_unit_activity(sae, activation_store, n_batches)

    k = round(protect_frac * sae.d_sae)
    if k <= 0:
        protected_idx = torch.empty(0, device=activity.device, dtype=torch.long)
    elif protect_mode == "top":
        protected_idx = torch.topk(activity, k=k, largest=True, sorted=False).indices
    elif protect_mode == "bottom":
        protected_idx = torch.topk(activity, k=k, largest=False, sorted=False).indices
    else:
        g = torch.Generator().manual_seed(seed)
        protected_idx = torch.randperm(sae.d_sae, generator=g)[:k].to(activity.device)

    mask = torch.zeros(sae.d_sae, device=device, dtype=torch.bool)
    mask[protected_idx] = True

    n_prot = int(mask.sum())
    print(f"[MaskedTrainer] Activity: mean={activity.mean():.6f} "
          f"max={activity.max():.6f} min={activity.min():.6f}; "
          f"{(fire_rate == 0).sum().item()} units never fired on the reference corpus")
    print(f"[MaskedTrainer] Protected {n_prot}/{sae.d_sae} units "
          f"({100 * n_prot / sae.d_sae:.1f}%) by protect_mode={protect_mode}, "
          f"{sae.d_sae - n_prot} units free to adapt.")

    stats = {
        "protect_frac": protect_frac, "protect_mode": protect_mode,
        "n_protected": n_prot, "n_free": sae.d_sae - n_prot,
        "activity_mean": activity.mean().item(),
        "activity_max": activity.max().item(),
        "n_never_fired": int((fire_rate == 0).sum()),
        "protected_fire_rate_mean": fire_rate[mask].mean().item() if n_prot else 0.0,
        "free_fire_rate_mean": fire_rate[~mask].mean().item(),
    }
    return mask, activity, stats


# ═════════════════════════════════════════════════════════════════════════
# Step B: the trainer
# ═════════════════════════════════════════════════════════════════════════

class MaskedTopKSAETrainer:
    """Fine-tunes a Top-K SAE while keeping protected units bit-exact.

    `protected_mask = None` (or an all-False mask) makes this a plain full
    fine-tune, which is how the `ftsae` / `scratchsae` arms are trained -- one
    code path for every arm, so the arms differ only in the mask.
    """

    def __init__(self, sae, activation_store, optimizer, scheduler, cfg,
                 protected_mask=None, device="cuda", freeze_b_dec=False,
                 normalize_decoder=True):
        self.sae = sae
        self.store = activation_store
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.normalize_decoder = normalize_decoder

        if protected_mask is None:
            protected_mask = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
        self.protected_mask = protected_mask.to(device)
        self.n_protected = int(self.protected_mask.sum())

        # Gradient masking alone is not enough: decoder renormalisation happens
        # outside the optimizer and would move protected columns. Keep immutable
        # copies and restore after every update, which makes "protected" exact.
        m = self.protected_mask
        self._frozen = {
            "W_enc": sae.W_enc.detach()[m, :].clone(),
            "b_enc": sae.b_enc.detach()[m].clone(),
            "W_dec": sae.W_dec.detach()[:, m].clone(),
        }
        # b_dec is shared by ALL units. Under Qwen-Scope's verified `raw`
        # encoding it does not enter the encoder, so protected units' *weights*
        # and pre-activations are exact either way; it does still shift the
        # shared decoder bias every unit's output is added to. Freezing it makes
        # the protected units' contribution to the reconstruction exact too, at
        # the cost of denying the new domain a decoder-bias shift. Default off,
        # matching patchsae.
        self.freeze_b_dec = freeze_b_dec or bool(m.all())
        if self.freeze_b_dec:
            self._frozen["b_dec"] = sae.b_dec.detach().clone()

        self.n_forward_passes_since_fired = torch.zeros(sae.d_sae, device=device)
        self.act_freq = torch.zeros(sae.d_sae, device=device)
        self.n_tokens_seen_for_freq = 0
        self.n_training_tokens = 0
        self.n_training_steps = 0
        self.history = []

        print(f"[MaskedTrainer] {self.n_protected} protected / "
              f"{sae.d_sae - self.n_protected} trainable units; "
              f"freeze_b_dec={self.freeze_b_dec}, "
              f"normalize_decoder={self.normalize_decoder}")

    # ------------------------------------------------------------- loss
    def _loss(self, x, dead_mask):
        """Normalised MSE + Top-K auxiliary dead-latent loss.

        The MSE normalisation (divide by the token's activation norm) is
        patchsae's, kept so loss curves are read the same way across the two
        experiments. `aux_k` is the Top-K replacement for ghost grads: let the
        `aux_k` most-promising *dead and free* units try to reconstruct the
        residual. Restricting revival to free units matters -- protected units
        were chosen as the highest-activity ones and are never dead in
        practice, so this changes nothing observable while keeping revival
        inside the trainable set.
        """
        sae = self.sae
        x_hat, acts, pre = sae(x)

        denom = x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
        mse = ((x_hat - x).pow(2) / denom).mean()

        aux = torch.tensor(0.0, device=x.device)
        if self.cfg.aux_k > 0 and dead_mask is not None:
            dead_free = dead_mask & (~self.protected_mask)
            n_dead = int(dead_free.sum())
            if n_dead > 0:
                resid = (x - x_hat).detach()
                k_aux = min(self.cfg.aux_k, n_dead)
                dead_idx = torch.nonzero(dead_free, as_tuple=True)[0]
                pre_dead = pre[:, dead_idx]
                vals, sel = torch.topk(pre_dead, k_aux, dim=-1)
                sparse = torch.zeros_like(pre_dead).scatter_(-1, sel, vals.relu())
                aux_hat = sparse @ sae.W_dec[:, dead_idx].t()
                aux = (aux_hat - resid).pow(2).sum(-1).mean() / \
                      resid.pow(2).sum(-1).mean().clamp_min(1e-6)
                aux = self.cfg.aux_coefficient * aux

        return x_hat, acts, {"mse": mse, "aux": aux, "loss": mse + aux}

    # ------------------------------------------------------- masking utils
    def _apply_gradient_mask(self):
        """Zero every gradient belonging to a protected unit."""
        if self.n_protected == 0:
            return
        m = self.protected_mask
        if self.sae.W_enc.grad is not None:
            self.sae.W_enc.grad[m, :] = 0.0
        if self.sae.b_enc.grad is not None:
            self.sae.b_enc.grad[m] = 0.0
        if self.sae.W_dec.grad is not None:
            self.sae.W_dec.grad[:, m] = 0.0
        if self.freeze_b_dec and self.sae.b_dec.grad is not None:
            self.sae.b_dec.grad.zero_()

    @torch.no_grad()
    def _remove_grad_parallel_to_decoder(self):
        """Project out the component of W_dec's gradient along each unit's own
        decoder direction (the standard trick: unit-norm columns make that
        component a no-op that only fights the renormalisation)."""
        W = self.sae.W_dec.data
        g = self.sae.W_dec.grad
        if g is None:
            return
        proj = (g * W).sum(dim=0, keepdim=True) * W
        g.sub_(proj)

    @torch.no_grad()
    def _restore_protected(self):
        if self.n_protected == 0 and not self.freeze_b_dec:
            return
        m = self.protected_mask
        if self.n_protected:
            self.sae.W_enc[m, :] = self._frozen["W_enc"]
            self.sae.b_enc[m] = self._frozen["b_enc"]
            self.sae.W_dec[:, m] = self._frozen["W_dec"]
        if "b_dec" in self._frozen:
            self.sae.b_dec.copy_(self._frozen["b_dec"])

    # ------------------------------------------------------------- step
    def _train_step(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        self.sae.train()

        dead_mask = (self.n_forward_passes_since_fired >
                     self.cfg.dead_feature_window).bool()
        x_hat, acts, losses = self._loss(x, dead_mask)

        with torch.no_grad():
            did_fire = (acts > 0).any(dim=0)
            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0
            self.act_freq += (acts > 0).float().sum(dim=0)
            self.n_tokens_seen_for_freq += acts.shape[0]

        losses["loss"].backward()
        self._apply_gradient_mask()
        if self.normalize_decoder:
            self._remove_grad_parallel_to_decoder()
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            if self.normalize_decoder:
                self.sae.normalize_decoder()
            self._restore_protected()

        return x_hat, acts, losses, dead_mask

    # -------------------------------------------------------------- fit
    def fit(self):
        cfg = self.cfg
        total = cfg.total_training_tokens
        pbar = tqdm(total=total, desc="masked-ft", unit="tok")
        t0 = time.time()

        # Bit-exactness check: verify the protected slice is untouched after the
        # first step, then again at the end. Cheap, and it is the whole claim.
        first_step_checked = False

        while self.n_training_tokens < total:
            x = self.store.get_batch_activations()
            x_hat, acts, losses, dead_mask = self._train_step(x)
            self.n_training_tokens += x.shape[0]
            self.n_training_steps += 1

            if not first_step_checked:
                self._assert_protected_intact("after step 1")
                first_step_checked = True

            if self.n_training_steps % cfg.log_every == 0:
                with torch.no_grad():
                    rec = self._step_metrics(x, x_hat, acts, losses, dead_mask)
                self.history.append(rec)
                pbar.set_description(
                    f"{self.n_training_steps}| mse {rec['mse']:.4f} "
                    f"aux {rec['aux']:.4f} fve {rec['fve']:.3f} l0 {rec['l0']:.0f} "
                    f"dead {rec['dead_frac']:.3f}")

            pbar.update(x.shape[0])

        pbar.close()
        self._assert_protected_intact("at end of training")
        self.wall_time = time.time() - t0
        print(f"[MaskedTrainer] {self.n_training_steps} steps, "
              f"{self.n_training_tokens} tokens, {self.wall_time / 60:.1f} min")
        return self.history

    @torch.no_grad()
    def _step_metrics(self, x, x_hat, acts, losses, dead_mask):
        err = (x - x_hat).pow(2).sum(-1).mean()
        var = (x - self.sae.b_dec).pow(2).sum(-1).mean()
        freq = self.act_freq / max(self.n_tokens_seen_for_freq, 1)
        m = self.protected_mask
        return {
            "step": self.n_training_steps,
            "tokens": self.n_training_tokens,
            "mse": losses["mse"].item(),
            "aux": float(losses["aux"]),
            "loss": losses["loss"].item(),
            "fve": (1 - err / var.clamp_min(1e-12)).item(),
            "l0": (acts > 0).float().sum(-1).mean().item(),
            "l0_protected": (acts[:, m] > 0).float().sum(-1).mean().item() if self.n_protected else 0.0,
            "l0_free": (acts[:, ~m] > 0).float().sum(-1).mean().item(),
            "dead_frac": (freq == 0).float().mean().item(),
            "n_dead_window": int(dead_mask.sum()),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _assert_protected_intact(self, when):
        if self.n_protected == 0:
            return
        m = self.protected_mask
        deltas = {
            "W_enc": (self.sae.W_enc[m, :] - self._frozen["W_enc"]).abs().max().item(),
            "b_enc": (self.sae.b_enc[m] - self._frozen["b_enc"]).abs().max().item(),
            "W_dec": (self.sae.W_dec[:, m] - self._frozen["W_dec"]).abs().max().item(),
        }
        worst = max(deltas.values())
        if worst != 0.0:
            raise RuntimeError(
                f"protected units moved {when}: max |delta| per tensor {deltas}")

    def save(self, path, extra=None):
        extra = dict(extra or {})
        extra.update({
            "protected_mask": self.protected_mask.cpu(),
            "n_protected": self.n_protected,
            "n_free": self.sae.d_sae - self.n_protected,
            "exact_protected_restore": True,
            "n_training_tokens": self.n_training_tokens,
            "n_training_steps": self.n_training_steps,
            "history": self.history,
        })
        out = self.sae.save(path, extra=extra)
        hist = Path(path).with_suffix(".history.json")
        json.dump(self.history, open(hist, "w"), indent=1)
        print(f"[MaskedTrainer] saved {out}")
        return out
