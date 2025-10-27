#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM = (
    "You are a careful quantitative estimator for Fermi-style questions.\n"
    "Provide brief reasoning if useful, but you MUST END with a single JSON object "
    'of the form {"L": <int>, "U": <int>} where L and U are INTEGERS (can be negative), '
    "representing a 99% confidence interval [10^L, 10^U]. If ≥99% of the mass is on exactly 10^x, "
    "use L=U=x. Do not include units in the JSON. Ensure L <= U. Keep the response less than 256 tokens."
)

def user_prompt(q: str, pct: float = 99.0) -> str:
    return (f"Question: {q}\n\n"
            f"Give a {pct:.2f}% confidence interval in the form [10^L, 10^U] with INTEGER L and U.\n"
            f'Finish with JSON ONLY: {{"L": <int>, "U": <int>}}.')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Qwen3 on Fermi-style questions (KV cache, batched).")
    p.add_argument("--base_model", type=str, default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Base"))
    p.add_argument("--adapter_dir", type=str, default=os.environ.get("ADAPTER_DIR", "./qwen3-4b-instruct-bf16-lora"),
                   help="Path to the initial/SFT LoRA adapter.")
    p.add_argument("--grpo_adapter_dir", type=str, default=os.environ.get("GRPO_ADAPTER_DIR", "./qwen3-4b-grpo"),
                   help="Path to the GRPO-updated LoRA adapter (output of training).")
    p.add_argument("--use_grpo", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, load the GRPO LoRA adapter instead of --adapter_dir.")
    p.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "./outputs"))
    p.add_argument("--debug", dest="debug", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, use data/fermi_test_20.txt; else data/fermi_test.txt.")
    p.add_argument("--raw", action="store_true", help="Evaluate the raw base model (ignore any LoRA adapter).")
    p.add_argument("--max_new", type=int, default=int(os.environ.get("MAX_NEW", "512")))
    p.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.0")))
    p.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "17")))
    p.add_argument("--trim_after_json", action=argparse.BooleanOptionalAction, default=True,
                   help="Cut response right after the FIRST JSON block.")
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "4")),
                   help="Batch size for evaluation generation.")
    p.add_argument("--flash", action=argparse.BooleanOptionalAction, default=True,
                   help="Request Flash-Attention 2 via attn_implementation=flash_attention_2 (fallback if unavailable).")
    p.add_argument("--ban_prefixes", nargs="*", default=os.environ.get("BAN_PREFIXES", "łazienk aimassage SpecWarn NdrFcShort").split(),
                   help="Space-separated list of strings to ban as generated prefixes.")
    return p.parse_args()

def set_seed(seed: int):
    try:
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

PAIR_RE = re.compile(r'^\s*(-?\d+)\s*$')
def read_eval_pairs(path: str) -> List[Tuple[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    pairs: List[Tuple[str, int]] = []
    i = 0
    while i + 1 < len(lines):
        q = lines[i]
        m = PAIR_RE.match(lines[i+1])
        if not m:
            raise ValueError(f"Expected integer on line {i+2}, got: {lines[i+1]!r}")
        pairs.append((q, int(m.group(1))))
        i += 2
    return pairs

# --- JSON parsing helpers (FIRST match, not last) ---
JSON_BLOCK_RE = re.compile(r'\{[^{}]*"L"\s*:\s*-?\d+[^{}]*"U"\s*:\s*-?\d+[^{}]*\}', re.DOTALL)
INT_RE = re.compile(r'"L"\s*:\s*(-?\d+).*?"U"\s*:\s*(-?\d+)', re.DOTALL)

def extract_LU(text: str) -> Tuple[int, int]:
    m = JSON_BLOCK_RE.search(text)  # FIRST match
    if not m:
        raise ValueError("No JSON {\"L\": int, \"U\": int} found.")
    block = m.group(0)
    m2 = INT_RE.search(block)
    if not m2:
        raise ValueError("JSON block found but no L/U ints parsed.")
    L = int(m2.group(1)); U = int(m2.group(2))
    if L > U: L, U = U, L
    return L, U

def first_json_end_index(text: str) -> Optional[int]:
    m = JSON_BLOCK_RE.search(text)
    return None if m is None else m.end()

# --- Stop / token helpers ---
def get_end_token_ids(tok) -> List[int]:
    ids: List[int] = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for t in ["<|im_end|>"]:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid not in (-1, None) and tid != tok.unk_token_id:
            ids.append(tid)
    out: List[int] = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out

def find_subseq(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(hay): return None
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i+n] == needle:
            return i
    return None

def build_bad_words_ids(tok, bad_starts: List[str]):
    ids = []
    for s in bad_starts:
        toks = tok(s, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks)
    return ids

LEADING_JUNK_PAT = re.compile(
    r"""^\s*(?:\ufeff)?(?:(?:łazienk\w*|aimassage|SpecWarn|NdrFcShort)[\s:–—-]*)+""",
    re.IGNORECASE | re.VERBOSE,
)
def sanitize_leading_noise(s: str) -> str:
    s2 = LEADING_JUNK_PAT.sub("", s)
    s2 = re.sub(r'^[^\x20-\x7E]+', '', s2)  # strip control/non-printables
    return s2.lstrip()

# --- Attention diagnostics ---
def inspect_attention_environment(model) -> Dict[str, Any]:
    """Return info about requested/actual attention backends."""
    info: Dict[str, Any] = {}

    cfg = getattr(model, "config", None)
    req = None
    for k in ("attn_implementation", "_attn_implementation"):
        if cfg is not None and hasattr(cfg, k):
            req = getattr(cfg, k)
            break
    info["requested_attn_implementation"] = req

    impls = set()
    has_attr_count = 0
    for m in model.modules():
        if hasattr(m, "attn_implementation"):
            has_attr_count += 1
            impls.add(getattr(m, "attn_implementation"))
    info["layer_attn_implementations"] = sorted(list(impls))
    info["layers_with_attn_attr"] = has_attr_count

    try:
        import flash_attn
        fav = getattr(flash_attn, "__version__", "unknown")
        info["flash_attn_package"] = True
        info["flash_attn_version"] = fav
    except Exception:
        info["flash_attn_package"] = False
        info["flash_attn_version"] = None

    sdp = {"flash_available": None, "mem_efficient_available": None, "math_available": None}
    try:
        from torch.backends.cuda import SDPBackend, sdp_kernel
        for name, backend in (("flash_available", SDPBackend.FLASH_ATTENTION),
                              ("mem_efficient_available", SDPBackend.EFFICIENT_ATTENTION),
                              ("math_available", SDPBackend.MATH)):
            try:
                sdp[name] = bool(sdp_kernel.is_kernel_available(backend))
            except Exception:
                sdp[name] = None
    except Exception:
        pass
    info["torch_sdp"] = sdp

    info["flash_attention2_active"] = (
        ("flash_attention_2" in info["layer_attn_implementations"]) or
        (info["requested_attn_implementation"] == "flash_attention_2" and info["flash_attn_package"])
    )

    return info

# --- Load model/tokenizer (KV cache ON, left padding) ---
def load_model_and_tokenizer(base_model: str,
                              adapter_dir: Optional[str],
                              use_adapter: bool,
                              try_flash: bool):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # critical for decoder-only batching

    model_kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,            # deprecation-safe
        device_map="auto",
        trust_remote_code=True,
    )
    if try_flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    if getattr(model.generation_config, "eos_token_id", None) is None and tok.eos_token_id is not None:
        model.generation_config.eos_token_id = tok.eos_token_id

    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    attn_info = inspect_attention_environment(model)
    print("\n[Attention diagnostics]")
    for k, v in attn_info.items():
        print(f"  - {k}: {v}")
    print()

    return tok, model, attn_info

# --- Batched generation (slice from max_input_len) ---
@torch.no_grad()
def generate_batch(
    tok, model, questions: List[str], max_new: int, temp: float, top_p: float,
    trim_after_json: bool, ban_prefixes: List[str]
) -> List[str]:
    msgs_list = []
    for q in questions:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(q, 99.0)},
        ]
        msgs_list.append(msgs)

    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)

    max_input_len = batch.input_ids.shape[1]

    eos_ids = get_end_token_ids(tok)
    user_turn_ids = tok("\n<|im_start|>user", add_special_tokens=False).input_ids
    bad_words_ids = build_bad_words_ids(tok, ban_prefixes) if ban_prefixes else None

    do_sample = (temp > 0.0 or top_p < 1.0)
    gen_kwargs = dict(
        max_new_tokens=max_new,
        return_dict_in_generate=True,
        output_scores=False,
        use_cache=True,
        do_sample=False,
    )
    if do_sample:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=max(temp, 1e-6),
            top_p=top_p,
        ))
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids

    out = model.generate(**batch, **gen_kwargs)
    seqs = out.sequences

    responses: List[str] = []
    for b in range(seqs.size(0)):
        gen_ids = seqs[b, max_input_len:].tolist()

        if eos_ids:
            cut = None
            for i, tid in enumerate(gen_ids):
                if tid in eos_ids:
                    cut = i
                    break
            if cut is not None:
                gen_ids = gen_ids[:cut]

        pos_user = find_subseq(gen_ids, user_turn_ids)
        if pos_user is not None:
            gen_ids = gen_ids[:pos_user]

        resp = tok.decode(gen_ids, skip_special_tokens=True).strip()
        resp = sanitize_leading_noise(resp)

        if trim_after_json:
            end_idx = first_json_end_index(resp)
            if end_idx is not None:
                resp = resp[:end_idx].rstrip()

        responses.append(resp)

    return responses

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    eval_file = "data/fermi_test_20.txt" if args.debug else "data/fermi_test.txt"
    split_tag = "20" if args.debug else "full"
    os.makedirs(args.out_dir, exist_ok=True)
    # Filenames reflect which adapter we used
    mode_tag = "raw" if args.raw else ("grpo" if args.use_grpo else "lora")
    out_json = os.path.join(args.out_dir, f"test_qwen_{mode_tag}_{split_tag}.json")
    out_metr = os.path.join(args.out_dir, f"metrics_{mode_tag}_{split_tag}.json")

    # Decide adapter path
    use_adapter = (not args.raw)
    chosen_adapter = None
    if use_adapter:
        chosen_adapter = args.grpo_adapter_dir if args.use_grpo else args.adapter_dir
        if not os.path.isdir(chosen_adapter):
            raise FileNotFoundError(
                f"Adapter directory not found: {chosen_adapter}. "
                f"Use --raw to eval base model or provide a valid path."
            )

    print(f"Eval file: {eval_file}")
    if args.raw:
        print("Mode: RAW base model")
    else:
        print(f"Mode: Base + LoRA ({'GRPO' if args.use_grpo else 'SFT/initial'}) -> {chosen_adapter}")
    print(f"Output: {out_json}")

    pairs = read_eval_pairs(eval_file)
    tok, model, attn_info = load_model_and_tokenizer(
        args.base_model,
        adapter_dir=chosen_adapter,
        use_adapter=use_adapter,
        try_flash=args.flash
    )

    results: List[Dict[str, Any]] = []
    hits = 0

    bs = max(1, int(args.batch_size))
    for start in range(0, len(pairs), bs):
        chunk = pairs[start:start+bs]
        qs = [q for (q, _) in chunk]
        ys = [y for (_, y) in chunk]

        resps = generate_batch(tok, model, qs, args.max_new, args.temp, args.top_p,
                               trim_after_json=args.trim_after_json, ban_prefixes=args.ban_prefixes)

        for (q, y), resp_text in zip(chunk, resps):
            try:
                L, U = extract_LU(resp_text)
            except Exception:
                L, U = None, None
            covered = (L is not None and U is not None and (L <= y <= U))
            hits += int(covered)

            results.append({
                "id": len(results)+1,
                "question": q,
                "response": resp_text,
                "true_answer": y,
                "lower": L if L is not None else None,
                "upper": U if U is not None else None,
            })
            print(f"[{len(results):03d}] y={y}  L={L}  U={U}  covered={covered}")

    coverage = hits / len(pairs) if pairs else 0.0
    print(f"\nCoverage ({mode_tag}-{split_tag}): {hits}/{len(pairs)} = {coverage:.3f}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metr_payload = {
        "coverage": coverage,
        "n": len(pairs),
        "mode": mode_tag,
        "split": split_tag,
        "attention": attn_info,
        "adapter_path": chosen_adapter if chosen_adapter else None,
        "base_model": args.base_model,
    }
    with open(out_metr, "w", encoding="utf-8") as f:
        json.dump(metr_payload, f, indent=2)

    print(f"Wrote: {out_json}\nWrote: {out_metr}")

if __name__ == "__main__":
    main()















