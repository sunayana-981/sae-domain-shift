#!/usr/bin/env python
"""Zero-shot vs LoRA fine-tuned accuracy on the DKSplit segmentation benchmark.

Arms
    zeroshot   Qwen/Qwen3.5-9B, no adapter
    lora       the same base + ABTdomain/dksplit-qwen-lora (PEFT, r128/alpha256)

Both arms share the tokenizer and chat template from the ADAPTER repo, the same
system prompt, greedy decoding and the same items, so the only difference
between them is the adapter. One arm per invocation, so the two can run
concurrently on separate GPUs; `compare_dksplit.py` does the paired statistics
afterwards from the per-item JSONL.

Metrics (the card's definitions):
    strict_em    prediction == truth
    lenient_em   prediction == truth, or == might_right where that is present
    mutation     prediction with spaces removed != input  (character
                 hallucination; the card reports 5.62% zero-shot -> 0.25%)

Both a raw and a normalised (casefold + collapsed whitespace) variant of each
EM are reported, because the card does not state its normalisation and the
choice is worth a few points.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS, RECONSTRUCTED, DEFAULT_PROMPT

# Optional: splice a Qwen-Scope SAE into the residual stream during generation.
try:
    from qsae.activations import ResidCapture, ResidIntervention, resolve_layer
    from qsae.sae import load_local_sae, load_qwenscope_sae
    _HAVE_QSAE = True
except Exception:
    _HAVE_QSAE = False

ADAPTER = "ABTdomain/dksplit-qwen-lora"
BASE = "Qwen/Qwen3.5-9B"
BENCH = "ABTdomain/dksplit-benchmark"
BENCH_FILES = {"5000": "benchmark_5000.csv", "1000": "sample_1000.csv"}

_THINK = re.compile(r".*?</think>", re.S)
_WS = re.compile(r"\s+")


def norm(s):
    """Casefold + collapse whitespace. Non-strings (a pandas NaN) become ""."""
    if not isinstance(s, str):
        return ""
    return _WS.sub(" ", s.strip()).casefold()


def clean_output(text):
    """Take the model's answer only.

    Qwen3.5's chat template carries thinking tags, so even with
    enable_thinking=False a stray `</think>` can appear; anything up to it is
    not the answer. Then keep the first non-empty line, since an untrained
    model frequently adds commentary after the result and scoring that as the
    prediction would understate zero-shot unfairly.
    """
    if "</think>" in text:
        text = _THINK.sub("", text, count=1)
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip()
        if line:
            return line
    return ""


def load_benchmark(which):
    path = hf_hub_download(BENCH, BENCH_FILES[which], repo_type="dataset",
                           token=False)
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    # Empty `might_right` cells must become real None, not a float NaN. Whether
    # `.where(..., None)` yields None or NaN depends on the pandas version and
    # the inferred column dtype, so convert explicitly: NaN is not None, so a
    # float would slip past every `alt is not None` guard downstream.
    df["might_right"] = [v if isinstance(v, str) and v.strip() else None
                         for v in df["might_right"]]
    return df


def build_model(arm, device, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dtype, device_map=device, token=False)
    if arm == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER, token=False)
        model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    return model, tok


THINK_BLOCK = "<think>\n\n</think>\n\n"


def render(tok, system, prefix, chat_mode="no_think"):
    """Render the chat prompt.

    The model card's snippet passes `enable_thinking=False`, but with the
    chat template shipped in the adapter repo that renders an *empty thinking
    block* -- `...assistant\n<think>\n\n</think>\n\n` -- which the adapter was
    evidently never trained on. Under it the fine-tuned model emits pure
    whitespace on most items (29% empty outputs, 2% accuracy). Removing the
    block so the prompt ends at `assistant\n` makes it answer correctly and
    immediately. Verified on the card's own examples:

        chatgptlogin   think_false -> ''               no_think -> 'chatgpt login'
        spotifywrapped think_false -> ' wrapped'       no_think -> 'spotify wrapped'
        mercibeaucoup  think_false -> ''               no_think -> 'merci beaucoup'

    Modes: no_think (default), think_false (the card's literal snippet),
    think_default (template default, which leaves `<think>` open).
    """
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": prefix}]
    if chat_mode == "think_default":
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    if chat_mode == "no_think":
        text = text.replace(THINK_BLOCK, "")
    return text


def stop_ids(tok):
    """Every id that should end a turn.

    `<|im_end|>` is the ChatML turn terminator and is NOT the tokenizer's
    `eos_token_id` here, so without this `generate` runs past the answer and
    loops: `'chatgpt login<|im_end|>\n\n\n chatgpt login<|im_end|>...'`.
    """
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def sae_fidelity(model, tok, sae, layer, prompts_text, prefixes, chat_mode,
                 device, batch=8):
    """Reconstruction quality on this task's own prompt activations.

    Measured with no hook registered, so the SAE cannot influence the
    activations it is scored on -- and measured on DKSplit prompts rather than
    a generic corpus, because that is the distribution the spliced run sees.
    """
    tok.padding_side = "left"
    err = var = cos_sum = 0.0
    n = 0
    for i in range(0, len(prefixes), batch):
        texts = [render(tok, prompts_text, p, chat_mode)
                 for p in prefixes[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with ResidCapture(model, layer) as cap:
            acts = cap.run(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"], use_cache=False)
        x = acts[enc["attention_mask"].bool()].float()
        x_hat, _, _ = sae(x)
        err += (x - x_hat).pow(2).sum(-1).sum().item()
        var += (x - sae.b_dec).pow(2).sum(-1).sum().item()
        cos_sum += F.cosine_similarity(x, x_hat, dim=-1).sum().item()
        n += x.shape[0]
    return dict(sae_fve=1 - err / max(var, 1e-12),
                sae_cosine=cos_sum / max(n, 1), sae_fidelity_tokens=n)


@torch.no_grad()
def run(model, tok, prompts_text, df, batch_size, max_new_tokens, device,
        chat_mode="no_think"):
    tok.padding_side = "left"
    eos = stop_ids(tok)
    preds, t0 = [], time.time()
    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        texts = [render(tok, prompts_text, r["prefix"], chat_mode) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=eos, use_cache=True)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(chunk, gen):
            preds.append(clean_output(tok.decode(g, skip_special_tokens=True)))
        if (i // batch_size) % 20 == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(rows)}  {rate:.1f} it/s  "
                  f"eta {(len(rows)-done)/max(rate,1e-9)/60:.1f}m", flush=True)
    return preds


def score(df, preds):
    recs, agg = [], dict(n=len(preds), strict_em=0, lenient_em=0,
                         strict_em_norm=0, lenient_em_norm=0,
                         mutation=0, empty=0)
    for r, p in zip(df.to_dict("records"), preds):
        truth, alt, pre = r["truth"], r["might_right"], r["prefix"]
        s_raw = p == truth
        l_raw = s_raw or (isinstance(alt, str) and p == alt)
        s_n = norm(p) == norm(truth)
        l_n = s_n or (isinstance(alt, str) and norm(p) == norm(alt))
        mut = p.replace(" ", "").casefold() != str(pre).replace(" ", "").casefold()
        agg["strict_em"] += s_raw; agg["lenient_em"] += l_raw
        agg["strict_em_norm"] += s_n; agg["lenient_em_norm"] += l_n
        agg["mutation"] += mut; agg["empty"] += (p == "")
        recs.append(dict(prefix=pre, truth=truth, might_right=alt, pred=p,
                         strict=bool(s_raw), lenient=bool(l_raw),
                         strict_norm=bool(s_n), lenient_norm=bool(l_n),
                         mutation=bool(mut)))
    n = max(agg["n"], 1)
    rates = {k: (v / n if k != "n" else v) for k, v in agg.items()}
    return rates, recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["zeroshot", "lora"])
    p.add_argument("--prompt", default=DEFAULT_PROMPT, choices=list(PROMPTS))
    p.add_argument("--benchmark", default="5000", choices=list(BENCH_FILES))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--chat_mode", default="no_think",
                   choices=["no_think", "think_false", "think_default"])
    p.add_argument("--sae_repo", default=None,
                   help="Splice this Qwen-Scope SAE into the residual stream "
                        "during generation, e.g. "
                        "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50. Omit for the "
                        "no-SAE condition.")
    p.add_argument("--sae_layer", type=int, default=-5,
                   help="Decoder block whose resid_post the SAE replaces "
                        "(negative counts from the end).")
    p.add_argument("--encode_variant", default="raw", choices=["raw", "centered"])
    p.add_argument("--sae_path", default=None,
                   help="Local SAE checkpoint (a fine-tuned arm) instead of "
                        "--sae_repo.")
    p.add_argument("--sae_label", default=None,
                   help="Name this SAE carries in the results, e.g. masked.")
    p.add_argument("--sae_fidelity_prompts", type=int, default=64)
    p.add_argument("--offset", type=int, default=0,
                   help="Skip the first N benchmark items. Use 2500 to score "
                        "only the half held out from SAE training.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="dksplit/out")
    args = p.parse_args()

    df = load_benchmark(args.benchmark)
    if args.offset:
        df = df.iloc[args.offset:].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    if args.sae_path:
        sae_tag = f"sae-{args.sae_label or Path(args.sae_path).stem}"
    elif args.sae_repo:
        sae_tag = f"sae{args.sae_layer}"
    else:
        sae_tag = "nosae"
    if args.offset:
        sae_tag += f"_off{args.offset}"
    tag = f"{args.arm}_{args.prompt}_{args.chat_mode}_{sae_tag}_bench{args.benchmark}" + \
          (f"_n{args.limit}" if args.limit else "")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(f"arm={args.arm}  prompt={args.prompt}  chat_mode={args.chat_mode}"
          f"{'  (RECONSTRUCTED, not the card text)' if args.prompt in RECONSTRUCTED else ''}")
    print(f"benchmark={BENCH_FILES[args.benchmark]}  n={len(df)}")
    print(f"ambiguous items with might_right: "
          f"{df['might_right'].notna().sum()} ({100*df['might_right'].notna().mean():.1f}%)")
    print("=" * 62, flush=True)

    model, tok = build_model(args.arm, args.device)

    sae, sae_info = None, {}
    if args.sae_repo or args.sae_path:
        if not _HAVE_QSAE:
            sys.exit("qsae package not importable; cannot splice an SAE")
        abs_layer = resolve_layer(model, args.sae_layer)
        if args.sae_path:
            sae = load_local_sae(args.sae_path, device=args.device,
                                 encode_variant=args.encode_variant)
            raw_cfg = {"base_model": f"local:{args.sae_path}"}
            print(f"[sae] local {args.sae_label or ''} {args.sae_path} "
                  f"layer {abs_layer}: d_in={sae.d_model} d_sae={sae.d_sae} "
                  f"k={sae.k}", flush=True)
        else:
            sae, raw_cfg = load_qwenscope_sae(args.sae_repo, abs_layer,
                                              device=args.device,
                                              encode_variant=args.encode_variant)
            print(f"[sae] {args.sae_repo} layer {abs_layer}: "
                  f"d_in={sae.d_model} d_sae={sae.d_sae} k={sae.k} "
                  f"(trained on {raw_cfg.get('base_model')})", flush=True)
        fid = sae_fidelity(model, tok, sae, args.sae_layer, PROMPTS[args.prompt],
                           list(df["prefix"].head(args.sae_fidelity_prompts)),
                           args.chat_mode, args.device)
        print(f"[sae] on this task's prompts: FVE {fid['sae_fve']:.4f}  "
              f"cosine {fid['sae_cosine']:.4f}  "
              f"({fid['sae_fidelity_tokens']} tokens)", flush=True)
        sae_info = dict(sae_repo=args.sae_repo or args.sae_path,
                        sae_label=args.sae_label or ("gsae" if args.sae_repo else None),
                        sae_layer=abs_layer,
                        sae_trained_on=raw_cfg.get("base_model"),
                        d_sae=sae.d_sae, k=sae.k,
                        encode_variant=args.encode_variant, **fid)

    t0 = time.time()
    if sae is None:
        preds = run(model, tok, PROMPTS[args.prompt], df, args.batch_size,
                    args.max_new_tokens, args.device, args.chat_mode)
    else:
        with ResidIntervention(model, args.sae_layer, sae=sae, mode="sae"):
            preds = run(model, tok, PROMPTS[args.prompt], df, args.batch_size,
                        args.max_new_tokens, args.device, args.chat_mode)
    rates, recs = score(df, preds)
    wall = time.time() - t0

    result = dict(tag=tag, arm=args.arm, prompt=args.prompt,
                  chat_mode=args.chat_mode,
                  prompt_is_reconstruction=args.prompt in RECONSTRUCTED,
                  base_model=BASE, adapter=ADAPTER if args.arm == "lora" else None,
                  benchmark=BENCH_FILES[args.benchmark], wall_seconds=wall,
                  offset=args.offset, **sae_info, **rates)
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    with open(out / f"{tag}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'':<18}{'raw':>9}{'normalised':>13}")
    print(f"{'strict EM':<18}{rates['strict_em']*100:8.2f}%{rates['strict_em_norm']*100:12.2f}%")
    print(f"{'lenient EM':<18}{rates['lenient_em']*100:8.2f}%{rates['lenient_em_norm']*100:12.2f}%")
    print(f"{'mutation rate':<18}{rates['mutation']*100:8.2f}%")
    print(f"{'empty output':<18}{rates['empty']*100:8.2f}%")
    print(f"\n{wall/60:.1f} min -> {out/f'{tag}.json'}")


if __name__ == "__main__":
    main()
