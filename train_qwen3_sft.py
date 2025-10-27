#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config (env overrides)
# -----------------------------
MODEL_NAME    = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Base")
DATA_PATH     = os.environ.get("DATA_PATH", "data/ci_xai_grok-4-fast-non-reasoning_99_train.json")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "./qwen3-4b-instruct-bf16-lora")
CONF_PCT      = float(os.environ.get("CONF_PCT", "99"))
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN", "4096"))

BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", "16"))
LR            = float(os.environ.get("LR", "1e-4"))
NUM_EPOCHS    = float(os.environ.get("NUM_EPOCHS", "2"))
SEED          = int(os.environ.get("SEED", "17"))

LORA_R        = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT  = float(os.environ.get("LORA_DROPOUT", "0.1"))

SAVE_STEPS    = int(os.environ.get("SAVE_STEPS", "25"))
LOG_STEPS     = int(os.environ.get("LOG_STEPS", "10"))
WARMUP_STEPS  = int(os.environ.get("WARMUP_STEPS", "10"))
WEIGHT_DECAY  = float(os.environ.get("WEIGHT_DECAY", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "0.5"))

# -----------------------------
# Prompt templates
# -----------------------------
SYSTEM_INSTR_TMPL = (
    "You are a careful quantitative estimator for Fermi-style questions.\n"
    "Provide brief reasoning if useful, but you MUST END with a single JSON object "
    'of the form {"L": <int>, "U": <int>} where L and U are INTEGERS (can be negative), '
    f"representing a {CONF_PCT}% confidence interval [10^L, 10^U]. If ≥{CONF_PCT}% of the mass is on exactly 10^x, "
    "use L=U=x. Do not include units in the JSON. Ensure L <= U. Keep the response less than 256 tokens."
)

def user_prompt(question: str, pct: float) -> str:
    return (
        f"Question: {question}\n\n"
        f"Give a {pct:.2f}% confidence interval in the form [10^L, 10^U] with INTEGER L and U.\n"
        'Finish with JSON ONLY: {"L": <int>, "U": <int>}.'
    )

# -----------------------------
# Tokenizer & Model (bf16)
# -----------------------------
def load_tokenizer_and_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # align pad to EOS for decoder-only
    # Right padding is fine for training with padding collator
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,   # A6000 supports bf16
        device_map="auto",
        trust_remote_code=True,
    )

    # Training-friendly settings
    model.config.use_cache = False  # needed with gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Align generation config with tokenizer
    gen = getattr(model, "generation_config", None)
    if gen is not None:
        if getattr(gen, "pad_token_id", None) is None:
            gen.pad_token_id = tok.pad_token_id
        if getattr(gen, "eos_token_id", None) is None and tok.eos_token_id is not None:
            gen.eos_token_id = tok.eos_token_id

    return tok, model

# -----------------------------
# PEFT LoRA
# -----------------------------
def wrap_with_lora(model):
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, peft_cfg)

# -----------------------------
# Dataset (expects JSON array with keys: question, response)
# -----------------------------
def load_raw_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")

# -----------------------------
# Serialization helpers
# -----------------------------
def build_texts(tokenizer, q: str, a: str) -> Dict[str, str]:
    """
    - prompt_text: system+user, ending right before assistant (add_generation_prompt=True)
    - full_text  : system+user+assistant content (we will ensure it ends with <|im_end|>)
    """
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_INSTR_TMPL},
        {"role": "user",   "content": user_prompt(q, CONF_PCT)},
    ]
    full_msgs = [
        {"role": "system", "content": SYSTEM_INSTR_TMPL},
        {"role": "user",   "content": user_prompt(q, CONF_PCT)},
        {"role": "assistant", "content": a},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text   = tokenizer.apply_chat_template(full_msgs,  tokenize=False, add_generation_prompt=False)
    return {"prompt_text": prompt_text, "full_text": full_text}

# -----------------------------
# Masking: supervise assistant including <|im_end|>
# -----------------------------
def mask_to_assistant_with_end(tokenizer, prompt_text: str, full_text: str, max_len: int) -> Dict[str, List[int]]:
    """
    Tokenize WITHOUT truncation, enforce <|im_end|> at the END of assistant,
    then left-crop to max_len while keeping the assistant span in labels.
    Labels cover [assistant_start ... <|im_end|>] (inclusive).
    """
    im_end_tok = "<|im_end|>"
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_tok)

    # 1) Tokenize w/o trunc to find true boundary and end marker presence
    prompt_nt = tokenizer(prompt_text, add_special_tokens=False)
    full_nt   = tokenizer(full_text,   add_special_tokens=False)
    pids_nt = prompt_nt["input_ids"]
    fids_nt = full_nt["input_ids"]

    # Prompt must be a prefix of full (by construction)
    if not (len(pids_nt) <= len(fids_nt) and pids_nt == fids_nt[:len(pids_nt)]):
        raise RuntimeError("Prompt is not a prefix of full serialization (unexpected template mismatch).")

    # 2) Ensure <|im_end|> is the LAST token; append if missing
    need_append = False
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        if len(fids_nt) == 0 or fids_nt[-1] != im_end_id:
            need_append = True
    else:
        # Fallback: if the tokenizer doesn't know the id, append the raw string and re-tokenize.
        need_append = (not full_text.rstrip().endswith(im_end_tok))

    if need_append:
        full_text = full_text + im_end_tok
        full_nt   = tokenizer(full_text, add_special_tokens=False)
        fids_nt   = full_nt["input_ids"]

    # 3) Compute truncation (left-crop to keep tail / assistant)
    total_len = len(fids_nt)
    drop = max(0, total_len - max_len)
    fids = fids_nt[drop:]
    prompt_len = max(0, len(pids_nt) - drop)

    attn = [1] * len(fids)
    labels = [-100] * len(fids)

    # 4) Locate the (first) <|im_end|> at/after assistant start in the CROPPED sequence
    #    If it's missing due to extreme truncation, we still supervise up to sequence end.
    end_pos = len(fids)
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        try:
            found = fids.index(im_end_id, prompt_len)
            end_pos = found + 1   # include the end token itself in loss
        except ValueError:
            pass
    else:
        # If we cannot map id, we can't reliably find it post-crop; just include to end.
        end_pos = len(fids)

    # 5) Label assistant span [prompt_len, end_pos)
    for i in range(prompt_len, min(end_pos, len(fids))):
        labels[i] = fids[i]
    #breakpoint()
    return {"input_ids": fids, "attention_mask": attn, "labels": labels}

# -----------------------------
# Map function
# -----------------------------
def to_masked_features(example: Dict[str, Any]) -> Dict[str, Any]:
    q = example.get("question", "")
    a = example.get("response", "")
    texts = build_texts(tokenizer, q, a)
    return mask_to_assistant_with_end(tokenizer, texts["prompt_text"], texts["full_text"], MAX_SEQ_LEN)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Speed hints
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    set_seed(SEED)

    tokenizer, base_model = load_tokenizer_and_model()
    model = wrap_with_lora(base_model)

    raw = load_raw_dataset(DATA_PATH).shuffle(seed=SEED)
    train_tok = raw.map(to_masked_features, batched=False, desc="Tokenizing & masking (assistant-only + end)")

    # Collator: strictly pad (no MLM, no label rewrite)
    collator = default_data_collator

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=bf16_ok,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,        # steps (not ratio)
        weight_decay=WEIGHT_DECAY,
        gradient_checkpointing=True,
        max_grad_norm=MAX_GRAD_NORM,
        report_to=[],                     # avoid legacy logging deps
        seed=SEED,
    )

    # Optional: inspect trainables once
    # try: model.print_trainable_parameters()
    # except Exception: pass

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_tok,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)      # saves LoRA adapter weights & adapter_config.json
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nDone. LoRA adapter saved to: {OUTPUT_DIR}")







