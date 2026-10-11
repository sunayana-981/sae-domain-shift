"""Residual-stream activation capture for Qwen3.5, and SAE splice hooks.

The vision experiment reads CLIP ViT block outputs through
`HookedVisionTransformer.run_with_cache` and serves them from a
`ViTActivationsStore`. The analogue here is a forward hook on
`model.model.layers[L]`, whose output is the post-block residual stream --
exactly Qwen-Scope's declared `resid_post` hook point.

Documents are packed into fixed-length windows rather than padded, so every
activation the store yields corresponds to a real token.
"""

import torch

from .data import CorpusSpec


def decoder_layers(model):
    """The decoder block list, however the LM happens to be wrapped.

    Walking `.model` / `.base_model` attributes does not work: on a
    `PreTrainedModel` whose base is itself, `base_model` returns `self`. So
    find it structurally instead -- the longest `ModuleList` of modules whose
    class name ends in "DecoderLayer" (or, failing that, simply the longest
    `ModuleList`). This is also what makes a merged PEFT model, an unmerged
    `PeftModel` and a bare `AutoModelForCausalLM` all resolve to the same
    blocks, which the eval matrix depends on.
    """
    best, best_n = None, -1
    for _, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList):
            continue
        kids = list(mod)
        if not kids:
            continue
        looks_like = type(kids[0]).__name__.endswith("DecoderLayer")
        score = len(kids) + (10_000 if looks_like else 0)
        if score > best_n:
            best, best_n = mod, score
    if best is None:
        raise AttributeError("could not locate a decoder layer list on this model")
    return best


def resolve_layer(model, layer):
    """Negative indices count from the end, as patchsae's block_layer does."""
    n = len(decoder_layers(model))
    return layer if layer >= 0 else n + layer


def _layer_out(out):
    return out[0] if isinstance(out, tuple) else out


class StopForward(Exception):
    """Raised by ResidCapture to abandon the forward pass once it has what it
    needs. Everything after the hooked block -- the remaining decoder layers
    and, more importantly, the 248k-wide `lm_head` -- is pure waste when all we
    want is one layer's residual stream, and the logits alone are ~2GB at the
    batch sizes used here."""


class ResidCapture:
    """Context manager that records `model.model.layers[L]`'s output.

    `stop_early=True` aborts the forward immediately after capturing. Callers
    must run the model inside `capture.run(...)`, or catch `StopForward`
    themselves.
    """

    def __init__(self, model, layer, stop_early=True):
        self.model = model
        self.block = decoder_layers(model)[resolve_layer(model, layer)]
        self.stop_early = stop_early
        self.acts = None
        self._h = None

    def __enter__(self):
        def hook(_mod, _inp, out):
            self.acts = _layer_out(out).detach()
            if self.stop_early:
                raise StopForward
        self._h = self.block.register_forward_hook(hook)
        return self

    def run(self, **kwargs):
        try:
            self.model(**kwargs)
        except StopForward:
            pass
        return self.acts

    def __exit__(self, *exc):
        self._h.remove()
        self._h = None


class ResidIntervention:
    """Substitutes something else for the layer's residual output.

    This is the LLM counterpart of patchsae's `SAEHookHandler`: it turns SAE
    reconstruction quality into a task-level number -- the cross-entropy the
    language model still achieves when it has to read the SAE's output instead
    of its own residual stream.

    mode="sae"   the SAE reconstruction (the arm under test)
    mode="zero"  zeros -- the destroy-everything floor that normalises
                 "CE recovered", following the sae-lens convention
    mode="mean"  the batch-mean activation vector, broadcast: a gentler floor
                 that keeps the layer's average output but no token-specific
                 information

    Position 0 is left untouched, because the SAE is never trained on
    position 0 (see `LMActivationStore.skip_first_token`) and a floor that
    clobbered it would not be comparable.

    That exemption applies only to a prefill pass. During cached generation
    each step's hidden state is [B, 1, d] for one *later* position, and
    treating index 0 of that as "position 0" would exempt every generated
    token and silently turn the intervention into a no-op for the entire
    continuation -- which is exactly the regime the generative task evals
    measure. So the exemption is conditioned on seq_len > 1.
    """

    def __init__(self, model, layer, sae=None, mode="sae", skip_first_token=True):
        if mode not in ("sae", "zero", "mean"):
            raise ValueError(f"mode must be sae/zero/mean, got {mode!r}")
        if mode == "sae" and sae is None:
            raise ValueError("mode='sae' needs an sae")
        self.block = decoder_layers(model)[resolve_layer(model, layer)]
        self.sae = sae
        self.mode = mode
        self.skip_first_token = skip_first_token
        self._h = None

    def _replacement(self, h):
        if self.mode == "sae":
            with torch.no_grad():
                recon, _, _ = self.sae(h.float())
            return recon.to(h.dtype)
        if self.mode == "zero":
            return torch.zeros_like(h)
        flat = h.reshape(-1, h.shape[-1]).float().mean(0)
        return flat.to(h.dtype).expand_as(h).clone()

    def __enter__(self):
        def hook(_mod, _inp, out):
            h = _layer_out(out)
            new = self._replacement(h)
            if self.skip_first_token and h.shape[1] > 1:
                new = torch.cat([h[:, :1], new[:, 1:]], dim=1)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        self._h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        self._h = None


def SAESplice(model, layer, sae, skip_first_token=True):
    """Backwards-compatible alias for the mode="sae" intervention."""
    return ResidIntervention(model, layer, sae=sae, mode="sae",
                             skip_first_token=skip_first_token)


class LatentIntervention:
    """Edits specific SAE latents in place, leaving the rest of the residual
    stream exactly as the model produced it.

    This is the LLM counterpart of the vision experiment's causal-steering
    edit: x' = x + W_D^T(z' - z). Unlike `ResidIntervention(mode="sae")`,
    which replaces the whole residual with the SAE's reconstruction (so an
    unmodified latent still picks up reconstruction error), this computes the
    delta from editing only the requested latent indices and adds *that* to
    the untouched residual -- an unmodified latent is an exact no-op, and any
    measured effect is attributable to the edited coordinates rather than to
    reconstruction noise. Positions are handled the same way as
    `ResidIntervention` (position 0 of a prefill pass exempted; every step of
    cached generation is edited).

    mode="ablate"  zero the requested latents
    mode="amplify" multiply the requested latents by `scale`
    """

    def __init__(self, model, layer, sae, latent_idx, mode="ablate", scale=1.0,
                 skip_first_token=True):
        if mode not in ("ablate", "amplify"):
            raise ValueError(f"mode must be ablate/amplify, got {mode!r}")
        self.block = decoder_layers(model)[resolve_layer(model, layer)]
        self.sae = sae
        self.latent_idx = torch.as_tensor(latent_idx, dtype=torch.long)
        self.mode = mode
        self.scale = scale
        self.skip_first_token = skip_first_token
        self._h = None

    def _edited_residual(self, h):
        with torch.no_grad():
            z, _ = self.sae.encode(h.float())  # TopKSAE.encode -> (acts, pre_acts)
            z_edit = z.clone()
            idx = self.latent_idx.to(z.device)
            if self.mode == "ablate":
                z_edit[..., idx] = 0.0
            else:
                z_edit[..., idx] = z_edit[..., idx] * self.scale
            delta = self.sae.decode(z_edit) - self.sae.decode(z)
        return (h.float() + delta).to(h.dtype)

    def __enter__(self):
        def hook(_mod, _inp, out):
            h = _layer_out(out)
            new = self._edited_residual(h)
            if self.skip_first_token and h.shape[1] > 1:
                new = torch.cat([h[:, :1], new[:, 1:]], dim=1)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        self._h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        self._h = None


# ═════════════════════════════════════════════════════════════════════════
# Token windows
# ═════════════════════════════════════════════════════════════════════════

class PackedTokenStream:
    """Streams a corpus and packs it into fixed-length token windows.

    Each window is `bos + seq_len-1` corpus tokens, documents separated by
    EOS. Packing (rather than per-document truncation + padding) keeps the
    token budget honest: `total_training_tokens` counts tokens the SAE
    actually saw.
    """

    def __init__(self, corpus: CorpusSpec, tokenizer, seq_len=512, seed=42,
                 split="train", min_doc_chars=32):
        self.corpus = corpus
        self.tok = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.split = split
        self.min_doc_chars = min_doc_chars
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.docs_seen = 0
        self._restarts = 0
        self._iter = self._windows()

    def _docs(self):
        """Yields text forever, re-opening the stream if it is exhausted."""
        while True:
            ds = self.corpus.stream(seed=self.seed + self._restarts, split=self.split)
            for ex in ds:
                txt = self.corpus.text_fn(ex)
                if txt and len(txt) >= self.min_doc_chars:
                    self.docs_seen += 1
                    yield txt
            self._restarts += 1

    def _windows(self):
        buf = []
        body = self.seq_len - (1 if self.bos is not None else 0)
        for txt in self._docs():
            buf.extend(self.tok(txt, add_special_tokens=False)["input_ids"])
            buf.append(self.eos)
            while len(buf) >= body:
                chunk, buf = buf[:body], buf[body:]
                yield ([self.bos] if self.bos is not None else []) + chunk

    def next_batch(self, n_seqs):
        return torch.tensor([next(self._iter) for _ in range(n_seqs)], dtype=torch.long)


class LMActivationStore:
    """Serves residual-stream activations as flat [n_tokens, d_model] batches.

    `token_source` is anything with `next_batch(n_seqs) -> LongTensor
    [n_seqs, seq_len]`: a `CachedTokenWindows` (what the experiment uses, so
    every arm reads identical tokens) or a live `PackedTokenStream`.

    `skip_first_token` drops position 0, whose residual norm is a large
    outlier in every decoder-only LM and would otherwise dominate both the MSE
    and the unit-activity estimate.
    """

    def __init__(self, model, token_source, layer, tokens_per_batch=4096,
                 seqs_per_forward=8, device="cuda", skip_first_token=True):
        self.model = model
        self.tokens = token_source
        self.layer = layer
        self.device = device
        self.tokens_per_batch = tokens_per_batch
        self.seqs_per_forward = seqs_per_forward
        self.skip_first_token = skip_first_token
        self._buf = []
        self._n_buf = 0
        self.tokens_served = 0

    @torch.no_grad()
    def _fill(self):
        ids = self.tokens.next_batch(self.seqs_per_forward).to(self.device)
        with ResidCapture(self.model, self.layer) as cap:
            acts = cap.run(input_ids=ids, use_cache=False)
        if self.skip_first_token:
            acts = acts[:, 1:]
        flat = acts.reshape(-1, acts.shape[-1]).float()
        self._buf.append(flat)
        self._n_buf += flat.shape[0]

    @torch.no_grad()
    def get_batch_activations(self):
        while self._n_buf < self.tokens_per_batch:
            self._fill()
        pool = torch.cat(self._buf, dim=0)
        out, rest = pool[: self.tokens_per_batch], pool[self.tokens_per_batch :]
        self._buf = [rest] if rest.shape[0] else []
        self._n_buf = rest.shape[0]
        self.tokens_served += out.shape[0]
        return out

    def batches(self, n):
        for _ in range(n):
            yield self.get_batch_activations()
