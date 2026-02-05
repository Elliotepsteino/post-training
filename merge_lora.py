#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model checkpoint.

Example:
    python merge_lora.py \
        --base-model /home/epsteine/post-training/model_weights/Qwen3-4B-Base \
        --lora-adapter /home/epsteine/post-training/qwen3-4b-instruct-bf16-lora \
        --output-dir /home/epsteine/post-training/model_weights/Qwen3-4B-Base-lora-merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        type=Path,
        help="Path to the base model checkpoint (e.g., model_weights/Qwen3-4B-Base).",
    )
    parser.add_argument(
        "--lora-adapter",
        required=True,
        type=Path,
        help="Path to the LoRA adapter directory produced by PEFT.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to save the merged full-precision weights.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to use while loading the model (default: bfloat16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading base model from {args.base_model} with dtype={torch_dtype} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {args.lora_adapter} ...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_adapter)

    print("Merging LoRA weights into the base model ...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir} ...")
    merged_model.save_pretrained(args.output_dir)

    print("Saving tokenizer to the merged checkpoint ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
