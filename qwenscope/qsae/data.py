"""Corpora for the Qwen-Scope masked-fine-tune experiment.

One reference corpus (the analogue of ImageNet in patchsae's vision
experiment: the distribution the pre-trained SAE is assumed to cover, and the
distribution whose active units get protected) plus three out-of-distribution
target corpora.

All four stream, so nothing but the tokenised window ever lands on disk.
`text_of` flattens each corpus's fields into one training string.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict

from datasets import load_dataset


@dataclass
class CorpusSpec:
    name: str
    repo_id: str
    kwargs: dict
    text_fn: Callable[[dict], str]
    role: str                       # "reference" | "ood"
    domain: str = ""
    note: str = ""
    shuffle_buffer: int = 1_000

    def stream(self, seed: int = 42, split: str = "train"):
        kw = dict(self.kwargs)
        kw.setdefault("split", split)
        ds = load_dataset(self.repo_id, streaming=True, **kw)
        return ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer)


def _join(*parts):
    return "\n\n".join(p.strip() for p in parts if p and str(p).strip())


CORPORA: Dict[str, CorpusSpec] = {
    # ── reference distribution: general educational web text ───────────────
    "fineweb": CorpusSpec(
        name="fineweb",
        repo_id="HuggingFaceFW/fineweb-edu",
        kwargs=dict(name="sample-10BT"),
        text_fn=lambda ex: ex["text"],
        role="reference",
        domain="general web",
        note="Stands in for ImageNet in the vision experiment: a broad "
             "pre-training-like distribution. NOTE: Qwen-Scope's paper states "
             "its SAEs were trained on undisclosed in-house pretraining data, "
             "so this is a PROXY for the SAE's training distribution, not the "
             "distribution itself. It is the same reference corpus SAE Boost "
             "(arXiv 2507.12990) uses for the identical purpose.",
    ),

    # ── three OOD target domains: standard Pile domain-shift subsets ───────
    # The Pile's per-domain components are the standard corpora for measuring
    # LM domain shift, and `ArmelR/the-pile-splitted` serves them as separate
    # streams. Each is paired with a generative benchmark in qsae/tasks.py, so
    # the same domain is measured both by reconstruction fidelity and by task
    # accuracy with the SAE spliced in.
    "pubmed_central": CorpusSpec(
        name="pubmed_central",
        repo_id="ArmelR/the-pile-splitted",
        kwargs=dict(name="PubMed Central"),
        text_fn=lambda ex: ex["text"],
        role="ood",
        domain="biomedical",
        note="Full-text biomedical papers. Task: PubMedQA.",
    ),
    "github": CorpusSpec(
        name="github",
        repo_id="ArmelR/the-pile-splitted",
        kwargs=dict(name="Github"),
        text_fn=lambda ex: ex["text"],
        role="ood",
        domain="source code",
        note="GitHub source files. Task: HumanEval pass@1.",
    ),
    # Two further OOD domains, chosen as *distribution shifts within a
    # capability* rather than as benchmark suites. Benchmarks like OSWorld or
    # VITA-Bench measure capability cliffs but are a few hundred eval items in
    # an interactive or multimodal setting -- there is nothing there to LoRA
    # fine-tune on, and this pipeline is text-only. These two are corpora of
    # millions of tokens whose token distribution is far from general web text.
    "lean": CorpusSpec(
        name="lean",
        repo_id="bigcode/the-stack-smol-xl",
        kwargs=dict(data_dir="data/lean"),
        text_fn=lambda ex: ex["content"],
        role="ood",
        domain="formal proofs (Lean)",
        note="Lean theorem-prover sources: tactic language, unicode maths, "
             "proof scripts. A shift *within* code capability -- the github "
             "corpus is imperative source, this is a dependently-typed proof "
             "language, so the two are not interchangeable.",
    ),
    "amharic": CorpusSpec(
        name="amharic",
        repo_id="HuggingFaceFW/fineweb-2",
        kwargs=dict(name="amh_Ethi"),
        text_fn=lambda ex: ex["text"],
        role="ood",
        domain="Amharic (Ethiopic script)",
        note="Low-resource non-Latin-script natural language. Maximally OOD "
             "in token distribution while remaining ordinary prose, and "
             "cross-lingual shift is what SAE Boost (arXiv 2507.12990) uses "
             "for this same experiment.",
    ),
    "freelaw": CorpusSpec(
        name="freelaw",
        repo_id="ArmelR/the-pile-splitted",
        kwargs=dict(name="FreeLaw"),
        text_fn=lambda ex: ex["text"],
        role="ood",
        domain="legal",
        note="US court opinions. Task: LegalBench definition_extraction.",
    ),
}

REFERENCE_CORPUS = "fineweb"
OOD_CORPORA = [n for n, s in CORPORA.items() if s.role == "ood"]


def get_corpus(name: str) -> CorpusSpec:
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; have {sorted(CORPORA)}")
    return CORPORA[name]


# ═════════════════════════════════════════════════════════════════════════
# Token cache
# ═════════════════════════════════════════════════════════════════════════
#
# Streaming these corpora live runs at ~2k tokens/s and occasionally drops a
# parquet connection, which would make every arm read *different* tokens and
# put the network on the critical path of a 9-run sweep. So each corpus is
# tokenised once into fixed-length windows on disk, and every arm reads the
# same windows: the train/eval split is a window-index split, identical across
# conditions, which is what makes the arms comparable at all.

import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch


def is_degenerate(text, min_compression_ratio=0.10, sample_chars=8192):
    """True for documents that are a few characters repeated forever.

    The Pile's GitHub subset contains large machine-generated data files -- one
    is literally the number 26 on every line for thousands of tokens. A base LM
    predicts those at a cross-entropy near zero, so a single such file dropped
    measured CE on the GitHub eval split from ~0.71 to ~0.18 and would have
    shrunk every arm's delta for reasons unrelated to sparse autoencoders.

    The measure is zlib compression ratio on the raw text, NOT unique-token
    ratio. The first version of this filter counted unique *tokens*, which made
    it tokenizer-dependent and wrong for low-resource scripts: Qwen's tokenizer
    has poor Ethiopic coverage, so Amharic prose becomes long runs of
    byte-level tokens, and the filter rejected 41% of the Amharic corpus --
    including ordinary Wikipedia articles -- while preferentially dropping
    longer documents (median 5,302 tokens dropped vs 1,346 kept). Compression
    ratio is a property of the text rather than of the tokenizer, and it
    separates the real thing cleanly:

        degenerate "26" file  0.057
        pubmed_central        0.188
        lean                  0.337
        amharic               0.455
        fineweb / freelaw     0.511

    so a 0.10 cut catches the data files and leaves every real corpus alone.
    """
    b = text[:sample_chars].encode("utf-8", errors="ignore")
    if len(b) < 256:
        return False
    return len(zlib.compress(b, 6)) / len(b) < min_compression_ratio


def build_token_cache(corpus: CorpusSpec, tokenizer, out_dir, n_tokens,
                      seq_len=512, seed=42, min_doc_chars=32, log_every=200_000,
                      min_compression_ratio=0.10):
    """Stream `corpus` until `n_tokens` tokens are packed, and save the windows.

    Windows are `[bos] + (seq_len-1)` corpus tokens with documents separated by
    EOS -- packing, not truncation + padding, so every cached token is real.
    Degenerate documents are dropped (see `is_degenerate`) and counted in the
    manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{corpus.name}.npy"
    man = out_dir / f"{corpus.name}.json"

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    body = seq_len - (1 if bos is not None else 0)
    n_windows = n_tokens // seq_len

    windows = np.empty((n_windows, seq_len), dtype=np.int32)
    buf, filled, docs, restarts, dropped = [], 0, 0, 0, 0

    while filled < n_windows:
        try:
            # Opening the stream is inside the try because the Hub answers 429
            # here, not only mid-iteration; an unauthenticated build of several
            # corpora reliably trips the rate limit.
            ds = corpus.stream(seed=seed + restarts)
            for ex in ds:
                txt = corpus.text_fn(ex)
                if not txt or len(txt) < min_doc_chars:
                    continue
                if is_degenerate(txt, min_compression_ratio):
                    dropped += 1
                    continue
                ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
                docs += 1
                buf.extend(ids)
                buf.append(eos)
                while len(buf) >= body and filled < n_windows:
                    chunk, buf = buf[:body], buf[body:]
                    row = ([bos] if bos is not None else []) + chunk
                    windows[filled] = row
                    filled += 1
                    if filled * seq_len % log_every == 0:
                        print(f"  [{corpus.name}] {filled * seq_len:,}/{n_tokens:,} tokens "
                              f"({docs:,} docs)", flush=True)
                if filled >= n_windows:
                    break
        except Exception as e:            # dropped parquet shard, or HTTP 429
            backoff = min(30 * (restarts + 1), 300)
            print(f"  [{corpus.name}] stream error ({type(e).__name__}: "
                  f"{str(e)[:80]}), retrying in {backoff}s", flush=True)
            time.sleep(backoff)
        restarts += 1
        if restarts > 20:
            raise RuntimeError(f"{corpus.name}: too many stream restarts")

    np.save(npy, windows)
    meta = dict(corpus=corpus.name, repo_id=corpus.repo_id, role=corpus.role,
                domain=corpus.domain, note=corpus.note, seq_len=seq_len,
                n_windows=int(n_windows), n_tokens=int(n_windows * seq_len),
                docs_consumed=docs, docs_dropped_degenerate=dropped,
                min_compression_ratio=min_compression_ratio,
                stream_restarts=restarts, seed=seed,
                tokenizer=tokenizer.name_or_path)
    json.dump(meta, open(man, "w"), indent=2)
    print(f"  [{corpus.name}] wrote {npy} {windows.shape} "
          f"({n_windows * seq_len:,} tokens from {docs:,} docs; "
          f"{dropped:,} degenerate docs dropped)", flush=True)
    return npy, meta


# Every corpus is read through the same window budget, because one of them is
# supply-limited: TimSchopf/medical_abstracts has 11,550 train documents
# (~3.15M tokens), so a cache larger than that repeats documents in its tail.
# Capping train+eval at 3.0M tokens keeps all three OOD corpora inside their
# unique supply *and* identical in size, so the arms are comparable and no arm
# trains on duplicated text.
DEFAULT_TRAIN_TOKENS = 2_700_000
DEFAULT_EVAL_TOKENS = 300_000


class CachedTokenWindows:
    """Serves cached token windows from a fixed, reproducible split.

    `split="train"` takes the first `train_tokens` worth of windows, `"eval"`
    the next `eval_tokens` worth. Both are index ranges fixed by the token
    budget rather than by the cache's size, so every corpus and every arm sees
    the same number of windows and the same disjoint split.

    BOTH splits are read in a seeded permutation. Shuffling eval is not
    cosmetic: packing is sequential within a document, so a contiguous slice of
    the eval range can be one or two long files. Scripts that consume only part
    of the split -- the LoRA eval reads 80 of 585 windows -- would then score
    the model on a couple of documents rather than on the corpus. Measured on
    the GitHub cache, the contiguous head gave mean CE 0.29 against 0.71 for a
    scattered sample of the same split. The permutation is seeded, so every arm
    still sees the identical sequence.
    """

    def __init__(self, cache_dir, corpus_name, split="train", seed=42,
                 train_tokens=DEFAULT_TRAIN_TOKENS,
                 eval_tokens=DEFAULT_EVAL_TOKENS):
        cache_dir = Path(cache_dir)
        self.path = cache_dir / f"{corpus_name}.npy"
        if not self.path.exists():
            raise FileNotFoundError(
                f"no token cache for {corpus_name!r} at {self.path}; "
                f"run scripts/build_token_cache.py first")
        self.meta = json.load(open(cache_dir / f"{corpus_name}.json"))
        self.windows = np.load(self.path, mmap_mode="r")
        self.corpus_name = corpus_name
        self.split = split
        self.seq_len = self.windows.shape[1]

        n = self.windows.shape[0]
        n_train = train_tokens // self.seq_len
        n_eval = eval_tokens // self.seq_len
        if n_train + n_eval > n:
            raise ValueError(
                f"{corpus_name} cache holds {n} windows "
                f"({n * self.seq_len:,} tokens) but the split needs "
                f"{n_train + n_eval} ({train_tokens + eval_tokens:,} tokens)")

        if split == "train":
            idx = np.arange(0, n_train)
            self.order = np.random.default_rng(seed).permutation(idx)
        elif split == "eval":
            idx = np.arange(n_train, n_train + n_eval)
            self.order = np.random.default_rng(seed + 1).permutation(idx)
        else:
            raise ValueError(f"split must be train/eval, got {split!r}")

        self.n_windows = len(self.order)
        self.n_tokens = self.n_windows * self.seq_len
        self._cursor = 0
        self.epochs = 0
    def next_batch(self, n_seqs):
        rows = []
        for _ in range(n_seqs):
            if self._cursor >= self.n_windows:
                self._cursor = 0
                self.epochs += 1
            rows.append(np.asarray(self.windows[self.order[self._cursor]]))
            self._cursor += 1
        return torch.from_numpy(np.stack(rows)).long()

    def reset(self):
        self._cursor = 0
        self.epochs = 0

    def __repr__(self):
        return (f"CachedTokenWindows({self.corpus_name}/{self.split}, "
                f"{self.n_windows} windows, {self.n_tokens:,} tokens)")
