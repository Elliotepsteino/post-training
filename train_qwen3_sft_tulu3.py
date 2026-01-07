#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from dataloader import load_year_mixture

# -----------------------------
# Default config (env overrides)
# -----------------------------
MODEL_NAME    = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Base")
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

SHARD_DIR     = os.environ.get("SHARD_DIR", "data_filtering/tulu_year_shards")
MAX_YEAR      = int(os.environ.get("MAX_YEAR", "2001"))
MAX_SAMPLES   = int(os.environ.get("MAX_SAMPLES", "0")) or None

# -----------------------------
# Prompt templates
# -----------------------------
def _build_system_instruction(pct: float) -> str:
    return (
        "You are a careful quantitative estimator for Fermi-style questions.\n"
        "Provide brief reasoning if useful, but you MUST END with a single JSON object "
        'of the form {"L": <int>, "U": <int>} where L and U are INTEGERS (can be negative), '
        f"representing a {pct}% confidence interval [10^L, 10^U]. If ≥{pct}% of the mass is on exactly 10^x, "
        "use L=U=x. Do not include units in the JSON. Ensure L <= U. Keep the response less than 256 tokens."
    )

SYSTEM_INSTR_TMPL = _build_system_instruction(CONF_PCT)

def user_prompt(question: str, pct: float) -> str:
    return (
        f"Question: {question}\n\n"
        f"Give a {pct:.2f}% confidence interval in the form [10^L, 10^U] with INTEGER L and U.\n"
        'Finish with JSON ONLY: {"L": <int>, "U": <int>}.'
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT on Tulu-3 year shards.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--shard-dir", default=SHARD_DIR, help="Directory containing year=YYYY.jsonl shards.")
    parser.add_argument("--max-year", type=int, default=MAX_YEAR, help="Use all shards up to and including this year.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Optional cap on training samples.")

    parser.add_argument("--conf-pct", type=float, default=CONF_PCT)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--num-epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)

    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--log-steps", type=int, default=LOG_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    return parser.parse_args()

# -----------------------------
# Tokenizer & Model (bf16)
# -----------------------------
def load_tokenizer_and_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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
# Serialization helpers
# -----------------------------
def build_texts(tokenizer, q: str, a: str) -> Dict[str, str]:
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

def mask_to_assistant_with_end(tokenizer, prompt_text: str, full_text: str, max_len: int) -> Dict[str, List[int]]:
    im_end_tok = "<|im_end|>"
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_tok)

    prompt_nt = tokenizer(prompt_text, add_special_tokens=False)
    full_nt   = tokenizer(full_text,   add_special_tokens=False)
    pids_nt = prompt_nt["input_ids"]
    fids_nt = full_nt["input_ids"]

    if not (len(pids_nt) <= len(fids_nt) and pids_nt == fids_nt[:len(pids_nt)]):
        raise RuntimeError("Prompt is not a prefix of full serialization (unexpected template mismatch).")

    need_append = False
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        if len(fids_nt) == 0 or fids_nt[-1] != im_end_id:
            need_append = True
    else:
        need_append = (not full_text.rstrip().endswith(im_end_tok))

    if need_append:
        full_text = full_text + im_end_tok
        full_nt   = tokenizer(full_text, add_special_tokens=False)
        fids_nt   = full_nt["input_ids"]

    total_len = len(fids_nt)
    drop = max(0, total_len - max_len)
    fids = fids_nt[drop:]
    prompt_len = max(0, len(pids_nt) - drop)

    attn = [1] * len(fids)
    labels = [-100] * len(fids)

    end_pos = len(fids)
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        try:
            found = fids.index(im_end_id, prompt_len)
            end_pos = found + 1
        except ValueError:
            pass

    for i in range(prompt_len, min(end_pos, len(fids))):
        labels[i] = fids[i]
    return {"input_ids": fids, "attention_mask": attn, "labels": labels}


class CausalLMPaddingCollator:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(field: str, pad_value: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.tensor(
                [
                    f[field] + [pad_value] * (max_len - len(f[field]))
                    for f in features
                ],
                dtype=dtype,
            )

        batch = {
            "input_ids": pad("input_ids", self.pad_token_id, torch.long),
            "attention_mask": pad("attention_mask", 0, torch.long),
            "labels": pad("labels", self.label_pad_token_id, torch.long),
        }
        return batch

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
    args = parse_args()

    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    SHARD_DIR = args.shard_dir
    MAX_YEAR = args.max_year
    MAX_SAMPLES = args.max_samples
    CONF_PCT = args.conf_pct
    MAX_SEQ_LEN = args.max_seq_len
    SYSTEM_INSTR_TMPL = _build_system_instruction(CONF_PCT)

    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    LR = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    SEED = args.seed

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout

    SAVE_STEPS = args.save_steps
    LOG_STEPS = args.log_steps
    WARMUP_STEPS = args.warmup_steps
    WEIGHT_DECAY = args.weight_decay
    MAX_GRAD_NORM = args.max_grad_norm

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(SEED)

    tokenizer, base_model = load_tokenizer_and_model()
    model = wrap_with_lora(base_model)

    raw = load_year_mixture(SHARD_DIR, MAX_YEAR, seed=SEED, max_samples=MAX_SAMPLES)
    train_tok = raw.map(to_masked_features, batched=False, desc="Tokenizing & masking (assistant-only + end)")

    collator = CausalLMPaddingCollator(tokenizer.pad_token_id, -100)

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args_train = TrainingArguments(
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
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        gradient_checkpointing=True,
        max_grad_norm=MAX_GRAD_NORM,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_tok,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nDone. LoRA adapter saved to: {OUTPUT_DIR}")
