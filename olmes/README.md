## Open Language Model Evaluation System (OLMES)

The OLMES (Open Language Model Evaluation System) repository is used within [Ai2](https://allenai.org)'s Open 
Language Model efforts to evaluate base and
instruction-tuned LLMs on a range of tasks. 

<details>
<summary>more details</summary>

The repository includes code to faithfully reproduce the evaluation results:

   * **OLMo 3:** TBD Title ([TBD Citation](...))
   * **OLMo 2:** 2 OLMo 2 Furious ([Team OLMo et al, 2024](https://arxiv.org/abs/2501.00656))
   * **TÜLU 3:** Pushing Frontiers in Open Language Model Post-Training ([Lambert et al, 2024](https://www.semanticscholar.org/paper/T/%22ULU-3%3A-Pushing-Frontiers-in-Open-Language-Model-Lambert-Morrison/5ca8f14a7e47e887a60e7473f9666e1f7fc52de7))
   * **OLMES:** A Standard for Language Model Evaluations ([Gu et al, 2024](https://www.semanticscholar.org/paper/c689c37c5367abe4790bff402c1d54944ae73b2a))
   * **OLMo:** Accelerating the Science of Language Models ([Groeneveld et al, 2024](https://www.semanticscholar.org/paper/ac45bbf9940512d9d686cf8cd3a95969bc313570))

The code base uses helpful features from the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 
by Eleuther AI, with a number of modifications and enhancements, including:

  * Support deep configurations for variants of tasks
  * Record more detailed data about instance-level predictions (logprobs, etc)
  * Custom metrics and metric aggregations
  * Integration with external storage options for results

</details>


### Setup

```sh
git clone https://github.com/allenai/olmes.git
cd olmes

# To install with uv:
uv sync
uv sync --group gpu # for vLLM support

# To install with pip:
pip install -e .
pip install -e ".[gpu]" # for vLLM support

# Activate the local venv (after it has been created once)
source /home/epsteine/post-training/olmes/.venv/bin/activate
```

Source the virtual environment whenever you open a new shell so `olmes`, `uv`, and the helper scripts resolve the correct dependencies.

### Workspace outputs (local setup)

To keep the repo tidy, all evaluation workspaces live under `/home/epsteine/post-training/workspaces/olmes` (create it once with `mkdir -p /home/epsteine/post-training/workspaces/olmes`). When running commands from `post-training/olmes`, point `--output-dir` at `../workspaces/olmes/<run_name>` (or rely on the helper scripts, which already use this layout).

## Usage

To evaluate a model and task (or task suite):

```bash
olmes \
    --model allenai/OLMo-2-0425-1B \
    --task arc_challenge::olmes \
    --output-dir workspace
```

This will launch the standard [OLMES](https://www.semanticscholar.org/paper/c689c37c5367abe4790bff402c1d54944ae73b2a) 
version of [ARC Challenge](https://www.semanticscholar.org/paper/88bb0a28bb58d847183ec505dda89b63771bb495) 
(which uses a curated 5-shot example, trying both multiple-choice and cloze formulations, and reporting
the max) with [OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B), storing the output in `workspace`.

### More options

**Multiple tasks** can be specified after `--task`:

```bash
olmes \
    --model allenai/OLMo-2-0425-1B \
    --task arc_challenge::olmes hellaswag::olmes \
    --output-dir workspace
```

**Inspect tasks.** You can sanity check using `--inspect`, which shows a sample prompt (and does 5-instance eval with a small `pythia`):

```bash
olmes --task arc_challenge:mc::olmes --inspect
```

**Dry run.** You can inspect the launch command with `--dry-run`:
```bash
olmes \
    --model allenai/OLMo-2-0425-1B \
    --task mmlu::olmes \
    --output-dir workspace \
    --dry-run
```

For a full list of arguments run `olmes --help`.


## Evaluating Qwen3 locally (ARC Challenge example)

To reproduce the internal Qwen3 ARC-Challenge runs, download the base checkpoint once and then point OLMES at the local path:

1. Install the Hugging Face hub client (needs one-time network access):
   ```bash
   python -m pip install --user huggingface_hub
   ```
2. (Optional but recommended) Authenticate for higher HF rate limits or to reach gated models such as `Qwen/Qwen3-4B-Instruct`:
   ```bash
   huggingface-cli login  # paste your HF token after accepting the model license
   ```
3. Download the base weights (adjust `LOCAL_DIR` as needed):
   ```bash
   export LOCAL_DIR=$PWD/../model_weights/Qwen3-4B-Base
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download("Qwen/Qwen3-4B-Base", local_dir="${LOCAL_DIR}")
   PY
   ```
4. Run the OLMES eval (this example mirrors the ARC-Challenge suite we used):
   ```bash
   uv run olmes \
       --model "${LOCAL_DIR}" \
       --task arc_challenge::olmes \
       --output-dir ../workspaces/olmes/workspace_qwen3
   ```

For the instruct checkpoint replace the repo id in step 3 with `Qwen/Qwen3-4B-Instruct` (login required) and reuse the same command in step 4, pointing `--model` at the instruct download.

## TÜLU-3 dev smoke test (limit=8)

We keep a convenience script in `/home/epsteine/post-training/run_tulu3_dev_limit8.sh` to sanity check a checkpoint on the TÜLU-3 dev suite:

```bash
cd /home/epsteine/post-training
./run_tulu3_dev_limit8.sh
```

Features:

- Uses all 5 local A6000s with a dynamic queue so each GPU stays busy.
- Applies `--limit 8` so each task evaluates on eight prompts (good for ~45 minutes runtime).
- Writes outputs to `/home/epsteine/post-training/workspaces/olmes/tulu3_dev_limit8/<task_alias>/` and logs under `olmes/logs/tulu3_dev_limit8/`.

To regenerate the high-level summary (consumed by the LaTeX report at `/home/epsteine/post-training/latex/main.tex`):

```bash
python - <<'PY'
import json, os
root = 'workspaces/olmes/tulu3_dev_limit8'
tasks = [
    ('gsm8k::tulu', 'gsm8k__tulu'),
    ('drop::llama3', 'drop__llama3'),
    ('minerva_math::tulu', 'minerva_math__tulu'),
    ('codex_humaneval::tulu', 'codex_humaneval__tulu'),
    ('codex_humanevalplus::tulu', 'codex_humanevalplus__tulu'),
    ('ifeval::tulu', 'ifeval__tulu'),
    ('popqa::tulu', 'popqa__tulu'),
    ('mmlu:mc::tulu', 'mmlu_mc__tulu'),
    ('alpaca_eval_v2::tulu', 'alpaca_eval_v2__tulu'),
    ('bbh:cot-v1::tulu', 'bbh_cot-v1__tulu'),
    ('truthfulqa::tulu', 'truthfulqa__tulu'),
]
summary = []
for alias, folder in tasks:
    metrics_path = os.path.join(root, folder, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        entry = next(t for t in data['tasks'] if t['alias'] == alias)
        summary.append({
            'alias': alias,
            'primary_score': entry['metrics']['primary_score'],
            'num_instances': entry['num_instances'],
        })
summary_path = os.path.join(root, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'wrote {summary_path}')
PY
```

Inspect `summary.json` or the per-task folders for the raw predictions, requests, and metrics.


## TÜLU-3 dev long run (limit=100)

For a deeper sweep of the TÜLU-3 dev suite (100 prompts per task), use `/home/epsteine/post-training/run_tulu3_dev_limit100.sh` after activating the virtualenv:

```bash
source /home/epsteine/post-training/olmes/.venv/bin/activate
cd /home/epsteine/post-training
chmod +x run_tulu3_dev_limit100.sh
./run_tulu3_dev_limit100.sh
```

The script fans tasks across all 5 local A6000s, writes results to `/home/epsteine/post-training/workspaces/olmes/tulu3_dev_limit${LIMIT}/`, and logs under `olmes/logs/tulu3_dev_limit${LIMIT}/`. Override the default limit by exporting `LIMIT` (for example, `LIMIT=50 ./run_tulu3_dev_limit100.sh`) to experiment with shorter or longer passes without editing the script.


## Running Eval Suites

We include full suites for our releases in [`task_suites.py`](oe_eval/configs/task_suites.py):

### Olmo 3 eval suite

To run all tasks specified in the [OLMo 3](.) technical report:

**Base Model Evaluation**

```bash
# Run the base easy evaluation (for evaluating small-scale experiments)
olmes \
    --model allenai/Olmo-3-1025-7B \
    --task \
        olmo3:base_easy:code_bpb \
        olmo3:base_easy:math_bpb \
        olmo3:base_easy:qa_rc \
        olmo3:base_easy:qa_bpb \
    --output-dir workspace

# Run the base main evaluation
olmes \
    --model allenai/Olmo-3-1025-7B \
    --task \
        olmo3:base:stem_qa_mc \
        olmo3:base:nonstem_qa_mc \
        olmo3:base:gen \
        olmo3:base:math \
        olmo3:base:code \
        olmo3:base:code_fim \
    --output-dir workspace

# Run the base held-out evaluation
olmes \
    --model allenai/Olmo-3-1025-7B \
    --task \
        olmo3:heldout \
    --output-dir workspace
```

**Instruct Model Evaluation**

```bash
# Run the instruct model evaluation
export OPENAI_API_KEY="..." # OpenAI models used for SimpleQA, AlpacaEval LLM Judge

olmes \
    --model allenai/Olmo-3-1025-7B \
    --task \
        olmo3:adapt \
    --output-dir workspace
```

<details>
<summary>Run safety evaluation</summary>

```bash
export OPEN_API_KEY="..." # An OpenAI API key
export HF_TOKEN="..." # Your huggingface token

oe-eval \
    --model hf-safety-eval \
    --task safety::olmo3 \
    --task-args '{ "generation_kwargs": { "max_gen_toks": 2048, "truncate_context": false } }' \
    --model-args '{"model_path":"allenai/Olmo-3-1025-7B", "max_length": 2048, "trust_remote_code": "true", "process_output": "r1_style"}'

# Note: For reasoning models, we use a max_gen_toks=32768, max_length=32768
```

</details>

### OLMo 2 eval suite

To run all tasks specified in the [OLMo 2](https://www.semanticscholar.org/paper/2-OLMo-2-Furious-OLMo-Walsh/685418141037ca44bb60b0b09d40fe521ec9e734) technical report:

```bash
olmes \
    --model allenai/OLMo-2-1124-7B \
    --task \
        core_9mcqa::olmes \
        mmlu:mc::olmes \
        olmo_2_generative::olmes \
        olmo_2_heldout::olmes \
    --output-dir workspace
```

### TÜLU 3 eval suite

To run all tasks used in the [TÜLU 3](https://www.semanticscholar.org/paper/T/%22ULU-3%3A-Pushing-Frontiers-in-Open-Language-Model-Lambert-Morrison/5ca8f14a7e47e887a60e7473f9666e1f7fc52de7) 
paper:

```bash
olmes \
    --model allenai/OLMo-2-1124-7B \
    --task \
        tulu_3_dev \
        tulu_3_unseen \
    --output-dir workspace
```


### OLMES: Standard 10 MCQA tasks

To run all 10 multiple-choice tasks from the [OLMES](https://www.semanticscholar.org/paper/c689c37c5367abe4790bff402c1d54944ae73b2a) paper:

```bash
olmes \
    --model allenai/OLMo-2-1124-7B \
    --task \
        core_9mcqa::olmes \
        mmlu::olmes \
    --output-dir workspace
```

### OLMo eval suite

To reproduce numbers in the [OLMo](https://www.semanticscholar.org/paper/ac45bbf9940512d9d686cf8cd3a95969bc313570) paper:

```bash
olmes \
    --model allenai/OLMo-7B-hf \
    --task \
        main_suite::olmo1 \
        mmlu::olmo1 \
    --output-dir workspace
```

## New Models / Tasks

<details>
<summary>Configure new models</summary>

### Model configuration

Models can be directly referenced by their Huggingface model path, e.g., `--model allenai/OLMoE-1B-7B-0924`,
or by their key in the [model library](oe_eval/configs/models.py), e.g., `--model olmoe-1b-7b-0924` which
can include additional configuration options (such as `max_length` for max context size and `model_path` for
local path to model).

The default model type uses the Huggingface model implementations, but you can also use the `--model-type vllm` flag to use
the vLLM implementations for models that support it, as well as `--model-type litellm` to run API-based models.

You can specify arbitrary JSON-parse-able model arguments directly in the command line as well, e.g.
```bash
olmes --model google/gemma-2b --model-args '{"trust_remote_code": true, "add_bos_token": true}' ...
```
To see a list of available models, run `oe-eval --list-models`, for a list of models containing a certain phrase,
you can follow this with a substring (any regular expression), e.g., `oe-eval --list-models llama`.

</details>

<details>
<summary>Configure new tasks</summary>

### Task configuration

To specify a task, use the [task library](oe_eval/configs/tasks.py) which have
entries like
```json
"arc_challenge:rc::olmes": {
    "task_name": "arc_challenge",
    "split": "test",
    "primary_metric": "acc_uncond",
    "num_shots": 5,
    "fewshot_source": "OLMES:ARC-Challenge",
    "metadata": {
        "regimes": ["OLMES-v0.1"],
    },
},
```
Each task can also have custom entries for `context_kwargs` (controlling details of the prompt),
`generation_kwargs` (controlling details of the generation), and `metric_kwargs` (controlling details of the metrics). The `primary_metric` indicates which metric field will be reported as the "primary score" for the task.

The task configuration parameters can be overridden on the command line, these will generally apply to all tasks, e.g.,
```bash
olmes --task arc_challenge:rc::olmes hellaswag::rc::olmes --split dev ...
```
but using a json format for each task, can be on per-task (but it's generally better to use the task 
library for this), e.g.,
```bash
olmes --task '{"task_name": "arc_challenge:rc::olmes", "num_shots": 2}' '{"task_name": "hellasag:rc::olmes", "num_shots": 4}' ...
```
For complicated commands like this, using `--dry-run` can be helpful to see the full command before running it.

To see a list of available tasks, run `oe-eval --list-tasks`, for a list of tasks containing a certain phrase,
you can follow this with a substring (any regular expression), e.g., `oe-eval --list-tasks arc`.

### Task suite configurations

To define a suite of tasks to run together, use the [task suite library](oe_eval/configs/task_suites.py),
with entries like:
```python
TASK_SUITE_CONFIGS["mmlu:mc::olmes"] = {
    "tasks": [f"mmlu_{sub}:mc::olmes" for sub in MMLU_SUBJECTS],
    "primary_metric": "macro",
}
```
specifying the list of tasks as well as how the metrics should be aggregated across the tasks.

</details>

## Evaluation output

Results are stored in the `--output-dir ...` directory. See [`OUTPUT_FORMATS.md`](OUTPUT_FORMATS.md) for more details. Other formats include:

- **Google Sheets** -- Save outputs to a Google Sheet by `--gsheet` (with authentication
stored in env var `GDRIVE_SERVICE_ACCOUNT_JSON`).
- **HF** -- Store in a Huggingface dataset directory by specifying `--hf-save-dir`
- **S3** -- Store in a remote directory (like `s3://...`) by specifying `--remote-output-dir`
- **Wandb** -- Store in a W&B project by specifying `--wandb-run-path`
