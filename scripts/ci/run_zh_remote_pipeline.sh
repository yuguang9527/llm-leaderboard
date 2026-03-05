#!/usr/bin/env bash
set -euo pipefail

# End-to-end Chinese leaderboard pipeline:
# 1) Submit remote Slurm vLLM job
# 2) Wait until job is running and API becomes healthy
# 3) Open SSH tunnel to local port
# 4) Run local zh evaluation and log to W&B

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_NAME="${1:-config-foxbrain-zh-tw.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="$ROOT_DIR/configs/$CONFIG_NAME"

SUNK_HOST="${SUNK_HOST:-sunk.cwb607-training.coreweave.app}"
SUNK_USER="${SUNK_USER:-rsong+cwb607}"
SUNK_SSH_KEY="${SUNK_SSH_KEY:-$HOME/.ssh/id_ed25519_sunk}"
REMOTE_VLLM_SCRIPT="${REMOTE_VLLM_SCRIPT:-/mnt/home/${SUNK_USER}/foxbrain_vllm_serve.sh}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
JOB_WAIT_TIMEOUT_SEC="${JOB_WAIT_TIMEOUT_SEC:-1800}"
API_WAIT_TIMEOUT_SEC="${API_WAIT_TIMEOUT_SEC:-3600}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-10}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
CANCEL_REMOTE_JOB_ON_EXIT="${CANCEL_REMOTE_JOB_ON_EXIT:-1}"

if [[ ! -f "$SUNK_SSH_KEY" ]]; then
  echo "ERROR: SSH key not found: $SUNK_SSH_KEY"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is required."
  exit 1
fi

if [[ -z "${OPENAI_COMPATIBLE_API_KEY:-}" ]]; then
  # vLLM API key can be a dummy string if server accepts arbitrary key.
  export OPENAI_COMPATIBLE_API_KEY="EMPTY"
fi

# Force public W&B endpoint for CI/CD runs.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config not found: $CONFIG_PATH"
  exit 1
fi

MODEL_ID_FROM_CONFIG="$(
  awk -F': ' '/pretrained_model_name_or_path:/{gsub(/"/, "", $2); print $2; exit}' "$CONFIG_PATH"
)"
if [[ -z "$MODEL_ID_FROM_CONFIG" ]]; then
  echo "ERROR: Failed to parse model.pretrained_model_name_or_path from $CONFIG_PATH"
  exit 1
fi

SSH_COMMON_ARGS=(
  -o StrictHostKeyChecking=no
  -o IdentitiesOnly=yes
  -i "$SUNK_SSH_KEY"
)

tunnel_pid=""
job_id=""
cleanup() {
  if [[ -n "$tunnel_pid" ]] && kill -0 "$tunnel_pid" >/dev/null 2>&1; then
    kill "$tunnel_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$CANCEL_REMOTE_JOB_ON_EXIT" == "1" && -n "$job_id" ]]; then
    ssh "${SSH_COMMON_ARGS[@]}" "${SUNK_USER}@${SUNK_HOST}" "scancel ${job_id}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "==> Submitting remote vLLM job via sbatch"
submit_output="$(
  ssh "${SSH_COMMON_ARGS[@]}" "${SUNK_USER}@${SUNK_HOST}" \
    "sbatch --export=ALL,MODEL_ID='${MODEL_ID_FROM_CONFIG}',PORT='${REMOTE_PORT}' ${REMOTE_VLLM_SCRIPT}"
)"
echo "$submit_output"
job_id="$(echo "$submit_output" | awk '/Submitted batch job/{print $4}')"
if [[ -z "$job_id" ]]; then
  echo "ERROR: Failed to parse Slurm job id."
  exit 1
fi
echo "Job ID: $job_id"
echo "Model ID: $MODEL_ID_FROM_CONFIG"

echo "==> Waiting for Slurm job RUNNING"
job_start_deadline="$(( $(date +%s) + JOB_WAIT_TIMEOUT_SEC ))"
remote_node=""
while true; do
  now="$(date +%s)"
  if (( now > job_start_deadline )); then
    echo "ERROR: Slurm job did not reach RUNNING state in time."
    exit 1
  fi

  status_line="$(ssh "${SSH_COMMON_ARGS[@]}" "${SUNK_USER}@${SUNK_HOST}" "squeue -h -j ${job_id} -o '%T %N'")"
  if [[ -z "$status_line" ]]; then
    echo "ERROR: Job disappeared from queue before RUNNING."
    exit 1
  fi

  state="$(echo "$status_line" | awk '{print $1}')"
  node="$(echo "$status_line" | awk '{print $2}')"
  echo "Job state: $state, node: $node"

  if [[ "$state" == "RUNNING" && -n "$node" && "$node" != "(null)" ]]; then
    remote_node="$node"
    break
  fi

  sleep "$POLL_INTERVAL_SEC"
done

echo "==> Opening SSH tunnel localhost:${LOCAL_PORT} -> ${remote_node}:${REMOTE_PORT}"
ssh -N "${SSH_COMMON_ARGS[@]}" -L "${LOCAL_PORT}:${remote_node}:${REMOTE_PORT}" "${SUNK_USER}@${SUNK_HOST}" &
tunnel_pid=$!

echo "==> Waiting for local vLLM endpoint readiness"
api_deadline="$(( $(date +%s) + API_WAIT_TIMEOUT_SEC ))"
until curl -fsS --connect-timeout 5 -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" "http://localhost:${LOCAL_PORT}/v1/models" >/dev/null; do
  now="$(date +%s)"
  if (( now > api_deadline )); then
    echo "ERROR: vLLM endpoint not ready before timeout."
    exit 1
  fi
  sleep "$POLL_INTERVAL_SEC"
done

echo "==> Running zh evaluation"
cd "$ROOT_DIR"
if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
if [[ "$SKIP_PIP_INSTALL" != "1" ]]; then
  python -m pip install --upgrade pip
  if [[ -f "scripts/ci/requirements-zh-runner.txt" ]]; then
    python -m pip install -r scripts/ci/requirements-zh-runner.txt
  else
    python -m pip install -r requirements.txt
  fi
fi
python -u scripts/run_zh_eval.py --config "$CONFIG_NAME"

echo "==> Pipeline completed successfully"
