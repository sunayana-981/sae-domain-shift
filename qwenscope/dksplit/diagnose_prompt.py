#!/usr/bin/env python
"""Raw generations from the adapter under different prompt renderings."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompts import PROMPTS

BASE, ADAPTER = "Qwen/Qwen3.5-9B", "ABTdomain/dksplit-qwen-lora"
CASES = ["chatgptlogin", "spotifywrapped", "mercibeaucoup"]
SYS = PROMPTS["advanced"]

tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                         device_map="cuda", token=False)
m = PeftModel.from_pretrained(m, ADAPTER, token=False).merge_and_unload(); m.eval()

def variants(prefix):
    msgs = [{"role":"system","content":SYS},{"role":"user","content":prefix}]
    v = {}
    v["think_false"] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    v["think_default"] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    v["no_think_block"] = v["think_false"].replace("<think>\n\n</think>\n\n", "")
    v["plain"] = f"{SYS}\n\nInput: {prefix}\nOutput:"
    v["user_only"] = tok.apply_chat_template([{"role":"user","content":SYS+"\n\n"+prefix}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return v

for prefix in CASES:
    print(f"\n{'='*66}\nINPUT: {prefix}")
    for name, text in variants(prefix).items():
        enc = tok(text, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
        new = out[0, enc["input_ids"].shape[1]:]
        with_sp = tok.decode(new, skip_special_tokens=False)
        print(f"  {name:16s} tail={text[-42:]!r}")
        print(f"  {'':16s} RAW={with_sp!r}")
