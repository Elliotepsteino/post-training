#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# Data reading (pairs: question, integer true_answer)
# -----------------------------
PAIR_RE = re.compile(r'^\s*(-?\d+)\s*$')

def read_pairs_txt(path: str) -> List[Tuple[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    pairs = []
    i = 0
    while i + 1 < len(lines):
        q = lines[i]
        m = PAIR_RE.match(lines[i + 1])
        if not m:
            raise ValueError(f"Expected integer on line {i+2}, got: {lines[i+1]!r}")
        pairs.append((q, int(m.group(1))))
        i += 2
    return pairs

# -----------------------------
# Prompt templates (ESCAPED BRACES!)
# -----------------------------
_SYSTEM_TMPL = (
    "You are a careful quantitative estimator for Fermi-style questions.\n"
    "Provide brief reasoning if useful, but you MUST END with a single JSON object "
    "of the form {{\"L\": <int>, \"U\": <int>}} where L and U are INTEGERS (can be negative), "
    "representing a {pct:.2f}% confidence interval [10^L, 10^U]. If ≥{pct:.2f}% of the mass is on exactly 10^x, "
    "use L=U=x. Do not include units in the JSON. Ensure L <= U. Keep the response less than 256 tokens."
)

def system_text(pct: float) -> str:
    return _SYSTEM_TMPL.format(pct=pct)

def user_prompt(question: str, pct: float) -> str:
    # Literal braces in JSON example are fine here.
    return (
        f"Question: {question}\n\n"
        f"Give a {pct:.2f}% confidence interval in the form [10^L, 10^U] with INTEGER L and U.\n"
        'Finish with JSON ONLY: {"L": <int>, "U": <int>}.'
    )

# -----------------------------
# JSON extract helpers (FIRST match)
# -----------------------------
JSON_BLOCK_RE = re.compile(r'\{[^{}]*"L"\s*:\s*-?\d+[^{}]*"U"\s*:\s*-?\d+[^{}]*\}', re.DOTALL)
INT_RE = re.compile(r'"L"\s*:\s*(-?\d+).*?"U"\s*:\s*(-?\d+)', re.DOTALL)

def parse_LU(text: str) -> Optional[Tuple[int,int]]:
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    b = m.group(0)
    m2 = INT_RE.search(b)
    if not m2:
        return None
    L = int(m2.group(1)); U = int(m2.group(2))
    if L > U: L, U = U, L
    return (L, U)

# -----------------------------
# Eval-style sanitization helpers
# -----------------------------
LEADING_JUNK_PAT = re.compile(
    r"""^\s*(?:\ufeff)?(?:(?:łazienk\w*|aimassage|SpecWarn|NdrFcShort)[\s:–—-]*)+""",
    re.IGNORECASE | re.VERBOSE,
)
def sanitize_leading_noise(s: str) -> str:
    s2 = LEADING_JUNK_PAT.sub("", s)
    s2 = re.sub(r'^[^\x20-\x7E]+', '', s2)  # strip control/non-printables
    return s2.lstrip()

# -----------------------------
# Winkler Score reward (with optional clipping)
# WS = (U-L) + (2/alpha) * | y - proj_[L,U](y) |
# We *maximize* reward => use negative WS, penalize invalid.
# -----------------------------
def winkler_reward(
    y: int,
    L: Optional[int],
    U: Optional[int],
    alpha: float,
    clip_abs: Optional[float] = 500.0,   # set None to disable clipping
    soft: bool = False                   # False = hard clip, True = soft clip
) -> float:
    """
    Winkler interval score:
      WS = (U - L) + (2/alpha) * | y - proj_[L,U](y) |
    We *maximize* reward => reward = -WS.
    Optional clipping keeps outliers from exploding gradients.
    """
    if L is None or U is None:
        # Make invalid strictly worse than any clipped valid output.
        base = (clip_abs if (clip_abs is not None and clip_abs > 0) else 500.0)
        return -2.0 * base

    if L > U:
        L, U = U, L

    width = U - L
    if y < L:
        miss = L - y
    elif y > U:
        miss = y - U
    else:
        miss = 0

    ws = width + (2.0 / alpha) * miss

    if clip_abs is not None and clip_abs > 0:
        if soft:
            ws = clip_abs * math.asinh(ws / clip_abs)
        else:
            ws = min(ws, clip_abs)

    return -float(ws)

# -----------------------------
# Token helpers
# -----------------------------
def get_eos_ids(tok) -> List[int]:
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    try:
        eid = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(eid, int) and eid not in (-1, None) and eid != tok.unk_token_id:
            ids.append(eid)
    except Exception:
        pass
    # unique
    out = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out

def find_first(hay: List[int], needles: List[int]) -> Optional[int]:
    for i, t in enumerate(hay):
        if t in needles:
            return i
    return None

# -----------------------------
# Batched generator (kept; used directly for rollouts)
# -----------------------------
@torch.no_grad()
def batched_generate_old(
    model, tok, questions: List[str], pct: float,
    max_new: int, temperature: float, top_p: float, top_k: Optional[int],
    bad_prefixes: Optional[List[str]] = None
) -> List[str]:
    msgs_list = []
    for q in questions[:1]:
        msgs_list.append([
            {"role": "system", "content": system_text(pct)},
            {"role": "user", "content": user_prompt(q, pct)}
        ])
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    input_lens = batch.attention_mask.sum(dim=1).tolist()

    eos_ids = get_eos_ids(tok)

    gen_kwargs = dict(
        max_new_tokens=max_new,
        return_dict_in_generate=True,
        output_scores=False,
        use_cache=True,
    )
    if temperature and temperature > 0:
        gen_kwargs.update(dict(do_sample=True, temperature=float(temperature)))
        if top_p is not None and 0 < top_p <= 1.0:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None and top_k > 0:
            gen_kwargs["top_k"] = int(top_k)
    else:
        gen_kwargs["do_sample"] = False
    #if False:
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    

    if bad_prefixes:
        bad_words_ids = []
        for s in bad_prefixes:
            bw = tok(s, add_special_tokens=False).input_ids
            if bw:
                bad_words_ids.append(bw)
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

    out = model.generate(**batch, **gen_kwargs)
    seqs = out.sequences

    responses = []
    for b in range(seqs.size(0)):
        start = input_lens[b]
        gen_ids = seqs[b, start:].tolist()
        if eos_ids:
            cut = find_first(gen_ids, eos_ids)
            if cut is not None:
                gen_ids = gen_ids[:cut]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        # Keep full text (eval-style), not just JSON
        #breakpoint()
        text = sanitize_leading_noise(text)
        # cut before a new user turn (eval-style guard)
        cut_pos = text.find("\n<|im_start|>user")
        if cut_pos != -1:
            text = text[:cut_pos].rstrip()
        responses.append(text)
    return responses

@torch.no_grad()
def batched_generate(
    model,
    tok,
    questions: List[str],
    pct: float,
    max_new: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    bad_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate per-question sequentially (no batching). Returns texts truncated right after {"L": int, "U": int}.
    """
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Collect EOS ids once
    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(int(tok.eos_token_id))
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end not in (-1, None) and im_end != tok.unk_token_id:
            eos_ids.append(int(im_end))
    except Exception:
        pass
    eos_ids = list(dict.fromkeys(eos_ids))  # unique

    # Build bad_words_ids once
    bad_words_ids = None
    if bad_prefixes:
        tmp = []
        for s in bad_prefixes:
            ids = tok(s, add_special_tokens=False).input_ids
            if ids:
                tmp.append(ids)
        if tmp:
            bad_words_ids = tmp

    # Sampling / stopping config
    do_sample = (temperature is not None and temperature > 0) or (top_p is not None and top_p < 1.0)
    base_gen_kwargs = dict(
        max_new_tokens=max_new,
        return_dict_in_generate=True,
        output_scores=False,
        use_cache=True,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    if do_sample:
        base_gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        base_gen_kwargs["top_p"] = float(top_p) if top_p is not None else 1.0
        if top_k is not None and top_k > 0:
            base_gen_kwargs["top_k"] = int(top_k)
    if eos_ids:
        base_gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    if bad_words_ids:
        base_gen_kwargs["bad_words_ids"] = bad_words_ids

    # Pattern for {"L": int, "U": int} with arbitrary spaces
    json_answer_pat = re.compile(r'\{\s*"L"\s*:\s*-?\d+\s*,\s*"U"\s*:\s*-?\d+\s*\}')

    outputs: List[str] = []
    for q in questions:
        msgs = [
            {"role": "system", "content": system_text(pct)},
            {"role": "user",   "content": user_prompt(q, pct)},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        enc = tok(prompt, return_tensors="pt").to(device)
        input_len = int(enc.attention_mask.sum().item())

        out = model.generate(**enc, **base_gen_kwargs)
        seq = out.sequences[0]

        gen_ids = seq[input_len:].tolist()
        if eos_ids:
            # stop at first EOS in generated span
            for i, t in enumerate(gen_ids):
                if t in eos_ids:
                    gen_ids = gen_ids[:i]
                    break

        text = tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        text = sanitize_leading_noise(text)
        cut_pos = text.find("\n<|im_start|>user")
        if cut_pos != -1:
            text = text[:cut_pos].rstrip()
        # drop any odd control glyphs
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]+', ' ', text).strip()

        # ---- NEW: truncate right after the first {"L": int, "U": int} ----
        m = json_answer_pat.search(text)
        if m:
            text = text[:m.end()].rstrip()
            # If you prefer to return only the JSON itself:
            # text = m.group(0)

        outputs.append(text)

    return outputs


# -----------------------------
# Build training minibatch for policy gradient
# inputs: concatenated (prompt + generated) token ids, padded LEFT
# targets: next tokens, ignore prompt part (mask=0) and pad positions with -100
# -----------------------------
def build_pg_minibatch(
    tok,
    prompts: List[str],
    generations: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    concat_ids = []
    masks = []
    for p, g in zip(prompts, generations):
        p_ids = tok(p, add_special_tokens=False).input_ids
        g_ids = tok(g, add_special_tokens=False).input_ids
        full = p_ids + g_ids + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
        mask = [0] * len(p_ids) + [1] * (len(full) - len(p_ids))
        concat_ids.append(full)
        masks.append(mask)

    max_len = max(len(x) for x in concat_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = []
    attn = []
    tgt = []
    for ids, m in zip(concat_ids, masks):
        pad_len = max_len - len(ids)
        inp = [pad_id] * pad_len + ids
        am = [0] * pad_len + [1] * len(ids)
        t = inp[1:] + [pad_id]
        label_mask = [0] * pad_len + m
        labels = []
        for tid, valid in zip(t, label_mask):
            labels.append(tid if valid == 1 else -100)

        input_ids.append(inp)
        attn.append(am)
        tgt.append(labels)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(tgt, dtype=torch.long),
    )

# -----------------------------
# Argparse
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("GRPO RL for Qwen3-4B on Fermi intervals (LoRA)")
    ap.add_argument("--base_model", type=str, default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Base"))
    ap.add_argument("--adapter_dir", type=str, default=os.environ.get("ADAPTER_DIR", "./qwen3-4b-instruct-bf16-lora/checkpoint-50"))
    ap.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "./qwen3-4b-grpo"))
    ap.add_argument("--debug", dest="debug", action=argparse.BooleanOptionalAction, default=False,
                    help="Use fermi_train_20.txt when true, else fermi_train.txt")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "1")))
    ap.add_argument("--device_batch_size", type=int, default=int(os.environ.get("DEVICE_BS", "2")),
                    help="Backward micro-batch for the PG step.")
    ap.add_argument("--examples_per_step", type=int, default=int(os.environ.get("EX_PER_STEP", "4")),
                    help="How many different questions per optimizer step.")
    ap.add_argument("--num_samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "4")),
                    help="Samples per question for rollouts.")
    ap.add_argument("--gen_microbatch_size", type=int, default=int(os.environ.get("GEN_MB", "8")),
                    help="How many prompts to decode in parallel per batched_generate call.")
    ap.add_argument("--max_new", type=int, default=int(os.environ.get("MAX_NEW", "512")))
    # Deterministic defaults; try TEMP≈0.7–0.9, TOP_P≈0.9–0.97, TOP_K>0 for exploration
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMP", "0.8")))
    ap.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "50")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "2e-5")))
    ap.add_argument("--weight_decay", type=float, default=float(os.environ.get("WD", "0.0")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "17")))
    ap.add_argument("--flash", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", "200")))
    ap.add_argument("--ban_prefixes", nargs="*", default=os.environ.get(
        "BAN_PREFIXES", "łazienk aimassage SpecWarn NdrFcShort").split())
    return ap.parse_args()

# -----------------------------
# Utility
# -----------------------------
def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data
    train_path = "data/fermi_train_20.txt" if args.debug else "data/fermi_train.txt"
    data = read_pairs_txt(train_path)
    print(f"Loaded {len(data)} train examples from {train_path}")

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # critical for decoder-only batching

    # Verify chat template has assistant slot
    try:
        tmp = tok.apply_chat_template(
            [{"role":"system","content":"x"},{"role":"user","content":"y"}],
            tokenize=False, add_generation_prompt=True
        )
        assert any(k in tmp for k in ["assistant", "<|assistant|>", "<|im_start|>assistant"]), \
            f"Tokenizer template missing assistant slot:\n{tmp[:400]}"
    except Exception as e:
        print(f"⚠️ Chat template sanity check skipped/failed: {e}")

    # Model (try Flash-Attn2 if requested)
    model_kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    # Load existing LoRA **as trainable**
    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(f"--adapter_dir not found: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True)

    # Some PEFT versions need explicit enabling
    if hasattr(model, "enable_adapter_layers"):
        try:
            model.enable_adapter_layers()
        except TypeError:
            try:
                model.enable_adapter_layers(True)
            except Exception:
                pass

    # Ensure KV cache and gen config pads/eos set
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tok.pad_token_id
        if getattr(model.generation_config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            model.generation_config.eos_token_id = tok.eos_token_id

    # Check trainable params
    n_trainable = count_trainable_params(model)
    print(f"Trainable params: {n_trainable:,}")
    if n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters found after loading the LoRA adapter.\n"
            "Ensure peft>=0.10 and `is_trainable=True`."
        )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.out_dir, exist_ok=True)

    conf_grid = [0.99]  #[0.90, 0.95, 0.99, 0.998]  # confidence levels
    alpha_map = {c: (1.0 - c) for c in conf_grid}

    step = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        i = 0
        while i < len(data):
            # One "optimizer step" processes `examples_per_step` examples
            examples = data[i : i + args.examples_per_step]
            i += args.examples_per_step
            if not examples:
                break

            # Sample confidence level once per step
            conf = random.choice(conf_grid)
            pct = conf * 100.0
            alpha = alpha_map[conf]

            # Prepare question strings & ground truth
            qs = [q for (q, _) in examples]
            ys = [y for (_, y) in examples]

            # Build prompts once (strings) matching batched_generate's internal construction
            # (We pass questions and pct to batched_generate directly.)

            # ============================
            # Rollouts via batched_generate (no pooled micro-batching)
            # We call it repeatedly to collect `num_samples` per question.
            # ============================
            model.eval()

            gens_by_ex: List[List[str]] = [[] for _ in range(len(qs))]
            for _ in range(args.num_samples):
                # generate one sample per question in a single batched call
                texts = batched_generate(
                    model, tok, qs, pct,
                    max_new=args.max_new,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=(args.top_k if args.top_k > 0 else None),
                    bad_prefixes=args.ban_prefixes
                )
                for ex_idx, txt in enumerate(texts):
                    gens_by_ex[ex_idx].append(txt)

            # Compute rewards from the FIRST JSON inside the full text
            all_generations: List[str] = []
            all_prompt_refs: List[str] = []
            all_rewards: List[float] = []

            # Build the exact prompts used inside batched_generate (for PG loss conditioning)
            prompts_for_pg: List[str] = []
            for q in qs:
                msgs = [
                    {"role": "system", "content": system_text(pct)},
                    {"role": "user", "content": user_prompt(q, pct)}
                ]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts_for_pg.append(prompt)

            for ex_idx, (q, y) in enumerate(zip(qs, ys)):
                p = prompts_for_pg[ex_idx]
                for gen_text in gens_by_ex[ex_idx]:
                    L_U = parse_LU(gen_text)
                    r = winkler_reward(y, *(L_U if L_U else (None, None)), alpha=alpha)
                    # Penalize completely empty strings a bit extra (rare, but can happen)
                    if not gen_text.strip():
                        r -= 5.0
                    all_generations.append(gen_text)
                    all_prompt_refs.append(p)
                    all_rewards.append(r)
            print(all_generations)
            print(all_rewards)
            B = len(all_generations)
            expected_B = len(qs) * args.num_samples
            assert B == expected_B, f"Unexpected rollout batch size B={B}, expected {expected_B}"

            # ============================
            # Policy Gradient step over all samples we just generated
            # ============================
            model.train()
            total_tokens = 0
            total_obj = 0.0

            # Advantage: (r - mean_r) per rollout (simple baseline)
            device_for_rewards = next(model.parameters()).device
            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device_for_rewards)
            adv = rewards_tensor - rewards_tensor.mean()

            # AMP context (CUDA only)
            if torch.cuda.is_available():
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            for start in range(0, B, args.device_batch_size):
                end = min(B, start + args.device_batch_size)
                mb_prompts = all_prompt_refs[start:end]
                mb_gens    = all_generations[start:end]
                mb_adv     = adv[start:end]

                inputs, attn, labels = build_pg_minibatch(tok, mb_prompts, mb_gens)
                inputs = inputs.to(model.device)
                attn   = attn.to(model.device)
                labels = labels.to(model.device)

                with autocast_ctx:
                    out = model(input_ids=inputs, attention_mask=attn, labels=None)
                    logits = out.logits[:, :-1, :]  # align to targets
                    targets = labels[:, 1:]         # next-token labels
                    mask = (targets != -100)

                    logprobs = F.log_softmax(logits, dim=-1)
                    tgt_lp = torch.gather(logprobs, dim=-1, index=targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                    tgt_lp = tgt_lp * mask.float()

                    token_counts = mask.sum(dim=1).clamp(min=1)
                    per_sample_obj = (tgt_lp.sum(dim=1) * mb_adv.to(tgt_lp.dtype))
                    per_sample_obj = per_sample_obj / token_counts

                    obj = per_sample_obj.mean()
                    loss = -obj

                loss.backward()
                total_obj += obj.item()
                total_tokens += int(token_counts.sum().item())

            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            model.zero_grad(set_to_none=True)

            print(f"[epoch {epoch}] step {step+1} | conf={pct:.1f}% | reward(mean)={rewards_tensor.mean().item():.4f} | obj={total_obj:.4f} | tokens/pg-batch={total_tokens}")
            step += 1

            # Save periodically
            if step % args.save_every == 0:
                save_dir = os.path.join(args.out_dir, f"checkpoint-{step}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tok.save_pretrained(save_dir)
                print(f"💾 Saved LoRA adapter to: {save_dir}")

    # Final save
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"\n✅ Done. LoRA (GRPO-updated) saved to: {args.out_dir}")

if __name__ == "__main__":
    main()










