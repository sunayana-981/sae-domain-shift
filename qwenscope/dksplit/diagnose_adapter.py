#!/usr/bin/env python
"""Why does the adapter arm score 2%? Compare unmerged vs merged application."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompts import PROMPTS
from eval_dksplit import render, clean_output

BASE, ADAPTER = "Qwen/Qwen3.5-9B", "ABTdomain/dksplit-qwen-lora"
CASES = ["chatgptlogin", "spotifywrapped", "mercibeaucoup", "brondsy", "robertdeniro"]
EXPECT = ["chatgpt login", "spotify wrapped", "merci beaucoup", "brondsy", "robert de niro"]

tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token

def gen(model, label):
    tok.padding_side = "left"
    texts = [render(tok, PROMPTS["advanced"], c) for c in CASES]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    print(f"\n--- {label}")
    for c, e, row in zip(CASES, EXPECT, out[:, enc["input_ids"].shape[1]:]):
        raw = tok.decode(row, skip_special_tokens=True)
        print(f"   {c:16s} -> {clean_output(raw)!r:34s} (want {e!r})")

print("loading base...", flush=True)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                            device_map="cuda", token=False)
base.eval()
print("arch:", type(base).__name__)
names = [n for n, _ in base.named_modules()]
print("has model.layers.0.mlp.down_proj:", "model.layers.0.mlp.down_proj" in names)
print("layer-0 linear names:", [n.split('.')[-1] for n in names
      if n.startswith("model.layers.0.") and isinstance(dict(base.named_modules())[n], torch.nn.Linear)])
print("layer-3 linear names:", [n.split('.')[-1] for n in names
      if n.startswith("model.layers.3.") and isinstance(dict(base.named_modules())[n], torch.nn.Linear)])
gen(base, "BASE (zero-shot)")

print("\nattaching adapter (unmerged)...", flush=True)
peft_model = PeftModel.from_pretrained(base, ADAPTER, token=False)
peft_model.eval()
n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A.default"))
print(f"adapted modules found: {n_lora}  (checkpoint has 128)")
print("active adapters:", peft_model.active_adapters)
gen(peft_model, "PEFT UNMERGED")

print("\nmerging...", flush=True)
merged = peft_model.merge_and_unload()
merged.eval()
gen(merged, "PEFT MERGED")
