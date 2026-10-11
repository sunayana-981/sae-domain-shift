"""Generative benchmarks scored with the SAE spliced into the residual stream.

This is the metric patchsae actually leads with -- `zeroshot_acc`, task accuracy
measured while the SAE's reconstruction replaces the model's own activations --
and it is what reconstruction fidelity (FVE, delta CE) does not directly tell
you. Each OOD corpus is paired with a benchmark from its own domain, plus GSM8K
as a general-capability retention check: a domain-adapted SAE should not degrade
reasoning it was never meant to touch.

    pubmed_central  ->  PubMedQA          (generate yes/no/maybe, exact match)
    github          ->  HumanEval         (generate a function body, pass@1)
    freelaw         ->  LegalBench        (definition_extraction, exact match)
    (general)       ->  GSM8K             (8-shot CoT, final-answer exact match)

All decoding is greedy, so a condition's score is deterministic and any
difference between arms comes from the intervention rather than from sampling.
"""

import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import load_dataset


# ═════════════════════════════════════════════════════════════════════════
# Generation
# ═════════════════════════════════════════════════════════════════════════

def _truncate(text, stops):
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


@torch.no_grad()
def greedy_generate(model, tok, prompts, max_new_tokens, stops, batch_size=8,
                    device="cuda"):
    """Batched greedy decoding with left padding.

    Left padding is required for batched generation: with right padding the
    pad tokens sit between the prompt and the first generated token, and the
    model continues from padding instead of from the prompt.
    """
    outs = []
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=1536).to(device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, use_cache=True)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                outs.append(_truncate(tok.decode(row, skip_special_tokens=True), stops))
    finally:
        tok.padding_side = old_side
    return outs


# ═════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═════════════════════════════════════════════════════════════════════════

def _norm(s):
    s = s.strip().lower()
    s = s.strip(string.punctuation + string.whitespace + "“”‘’")
    return " ".join(s.split())


_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _last_number(text):
    m = _NUM.findall(text.replace(",", ""))
    if not m:
        return None
    v = m[-1].replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(v)
    except ValueError:
        return None


def _run_python(source, timeout=12):
    """Execute one HumanEval candidate in a separate process.

    HumanEval's pass@1 is defined by actually running the generated function
    against the benchmark's unit tests, so this runs model-written code. Each
    candidate goes in its own short-lived subprocess with a hard timeout, so a
    hang or a crash costs one problem rather than the run.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        Path(path).unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════════════
# Task definitions
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class GenTask:
    name: str
    domain: str                  # corpus this is the target task for, or "general"
    metric: str
    max_new_tokens: int
    stops: list
    build: Callable              # (n, seed) -> list of {"prompt", "ref", ...}
    score: Callable              # (item, completion) -> bool
    batch_size: int = 8
    n_default: int = 200


# ── PubMedQA ────────────────────────────────────────────────────────────
def _pubmedqa_build(n, seed):
    ds = list(load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=rng).tolist()
    shots, rest = idx[:3], idx[3 : 3 + n]

    def block(ex, with_answer=True):
        ctx = " ".join(ex["context"]["contexts"])[:1500]
        s = (f"Abstract: {ctx}\nQuestion: {ex['question']}\n"
             f"Answer (yes, no, or maybe):")
        return s + f" {ex['final_decision']}\n\n" if with_answer else s

    prefix = "".join(block(ds[i]) for i in shots)
    return [{"prompt": prefix + block(ds[i], False), "ref": ds[i]["final_decision"]}
            for i in rest]


def _pubmedqa_score(item, comp):
    w = _norm(comp).split()
    return bool(w) and w[0] == item["ref"].strip().lower()


# ── HumanEval ───────────────────────────────────────────────────────────
def _humaneval_build(n, seed):
    ds = list(load_dataset("openai/openai_humaneval", split="test"))
    return [{"prompt": ex["prompt"], "ref": ex["test"],
             "entry_point": ex["entry_point"]} for ex in ds[:n]]


def _humaneval_score(item, comp):
    prog = (item["prompt"] + comp + "\n\n" + item["ref"] +
            f"\n\ncheck({item['entry_point']})\n")
    return _run_python(prog)


# ── LegalBench: definition_extraction ───────────────────────────────────
def _legal_build(n, seed):
    try:
        shots_ds = list(load_dataset("nguha/legalbench", "definition_extraction",
                                     split="train"))
    except Exception:
        shots_ds = []
    ds = list(load_dataset("nguha/legalbench", "definition_extraction",
                           split="test"))
    if not shots_ds:
        shots_ds, ds = ds[:4], ds[4:]

    def block(ex, with_answer=True):
        s = (f"Sentence: {ex['text'].strip()}\n"
             f"The term being defined in this sentence is:")
        return s + f" {ex['answer']}\n\n" if with_answer else s

    prefix = "".join(block(ex) for ex in shots_ds[:4])
    return [{"prompt": prefix + block(ex, False), "ref": ex["answer"]}
            for ex in ds[:n]]


def _legal_score(item, comp):
    return _norm(comp) == _norm(item["ref"])


# ── GSM8K (general-capability retention) ────────────────────────────────
def _gsm8k_build(n, seed):
    train = list(load_dataset("openai/gsm8k", "main", split="train"))
    test = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng = torch.Generator().manual_seed(seed)
    shots = torch.randperm(len(train), generator=rng).tolist()[:8]
    tidx = torch.randperm(len(test), generator=rng).tolist()[:n]

    def block(ex, with_answer=True):
        s = f"Question: {ex['question'].strip()}\nAnswer:"
        if not with_answer:
            return s
        reasoning, _, final = ex["answer"].partition("####")
        reasoning = re.sub(r"<<.*?>>", "", reasoning).strip()
        return f"{s} {reasoning} The answer is {final.strip()}.\n\n"

    prefix = "".join(block(train[i]) for i in shots)
    return [{"prompt": prefix + block(test[i], False),
             "ref": test[i]["answer"].split("####")[-1].strip()}
            for i in tidx]


def _gsm8k_score(item, comp):
    got = _last_number(comp)
    want = _last_number(item["ref"])
    return got is not None and want is not None and abs(got - want) < 1e-4


TASKS = {
    "pubmedqa": GenTask(
        name="pubmedqa", domain="pubmed_central", metric="exact_match",
        max_new_tokens=8, stops=["\n", "Abstract:", "Question:"],
        build=_pubmedqa_build, score=_pubmedqa_score,
        batch_size=8, n_default=300),
    "humaneval": GenTask(
        name="humaneval", domain="github", metric="pass@1",
        max_new_tokens=384,
        stops=["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(",
               "\n@", '\n"""', "\nassert "],
        build=_humaneval_build, score=_humaneval_score,
        batch_size=8, n_default=164),
    "legalbench_defext": GenTask(
        name="legalbench_defext", domain="freelaw", metric="exact_match",
        max_new_tokens=16, stops=["\n", "Sentence:"],
        build=_legal_build, score=_legal_score,
        batch_size=8, n_default=300),
    "gsm8k": GenTask(
        name="gsm8k", domain="general", metric="exact_match",
        max_new_tokens=256, stops=["\nQuestion:", "\n\n"],
        build=_gsm8k_build, score=_gsm8k_score,
        batch_size=8, n_default=200),
}


def run_task(task: GenTask, model, tok, n=None, seed=42, device="cuda",
             keep_completions=0):
    items = task.build(n or task.n_default, seed)
    comps = greedy_generate(model, tok, [it["prompt"] for it in items],
                            task.max_new_tokens, task.stops,
                            batch_size=task.batch_size, device=device)
    hits = [bool(task.score(it, c)) for it, c in zip(items, comps)]
    out = {"task": task.name, "metric": task.metric, "domain": task.domain,
           "n": len(items), "accuracy": sum(hits) / max(len(hits), 1)}
    if keep_completions:
        out["samples"] = [{"ref": str(it["ref"])[:120], "completion": c[:400],
                           "correct": h}
                          for it, c, h in list(zip(items, comps, hits))[:keep_completions]]
    return out
