#!/usr/bin/env bash
# run_verl_1gpu_fsdp.sh — VERL GRPO + LoRA on 1 GPU (FSDP actor, vLLM TP=1)

set -euo pipefail

############################
# User-editable defaults  ##
############################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Data (parquet already created with a `ground_truth` column)
TRAIN_PARQUET="${TRAIN_PARQUET:-$PWD/data/fermi_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PWD/data/fermi_test.parquet}"

# Base model + reward function
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Base}"
REWARD_PY="${REWARD_PY:-$PWD/reward_fermi.py}"  # must define: compute_score(data_source, solution_str, ground_truth, extra_info=None)

# Training knobs
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-2}"

# LoRA
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-all-linear}"

# Rollout engine (vLLM on 1 GPU)
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-vllm}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.6}"
ROLLOUTS_PER_PROMPT="${ROLLOUTS_PER_PROMPT:-6}"

# Logging
WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_fermi}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_lora_grpo}"

# Training schedule
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-200}"
TEST_FREQ="${TEST_FREQ:-50}"

############################
# Light sanity checks     ##
############################
echo "==> Using GPUs: ${CUDA_VISIBLE_DEVICES}"
python - <<'PY'
import importlib, sys
try:
    importlib.import_module("verl")
    print("✅ VERL import OK")
except Exception as e:
    print("❌ VERL import failed:", e)
    sys.exit(1)
PY

for p in "$TRAIN_PARQUET" "$VAL_PARQUET" "$REWARD_PY"; do
  if [[ ! -s "$p" ]]; then
    echo "❌ Missing or empty: $p" >&2
    exit 1
  fi
done

# Helpful env for stability
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0

echo "==> Rollout engine: ${ROLLOUT_ENGINE}"
echo "==> Train parquet : ${TRAIN_PARQUET}"
echo "==> Val parquet   : ${VAL_PARQUET}"
echo "==> Reward script : ${REWARD_PY}"
echo "==> Base model    : ${BASE_MODEL}"
echo "==> W&B project   : ${WANDB_PROJECT} / ${EXPERIMENT_NAME}"

# === FSDP on 1 GPU, vLLM rollout on same GPU ===
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_multi_modal_inputs=False \
  data.reward_fn_key=ground_truth \
  \
  custom_reward_function.path="${REWARD_PY}" \
  custom_reward_function.name=compute_score \
  \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.torch_dtype="bfloat16" \
  \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.optim.lr=2e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules="${LORA_TARGETS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  \
  actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${VLLM_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUTS_PER_PROMPT}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
  \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}"
















