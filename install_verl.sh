# 1) Fresh env (recommended)
conda create -y -n verl python=3.10
conda activate verl

# 2) Install a matching PyTorch + CUDA (example: CUDA 12.x)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

# 3) Clone VERL and install from source (best path)
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
# (Optional rollout backends)
pip install vllm>=0.8.2  # avoid vLLM 0.7.x
# Optional extras you may use
pip install transformers peft accelerate datasets sentencepiece wandb