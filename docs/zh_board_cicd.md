# ZH Board CI/CD

This pipeline automates Chinese leaderboard evaluation end-to-end:

1. Submit remote vLLM Slurm job on CoreWeave.
2. Wait until job is `RUNNING`.
3. Create SSH tunnel to remote vLLM OpenAI-compatible endpoint.
4. Run local `scripts/run_zh_eval.py`.
5. Log results to W&B project configured in your model config.

## Local one-command run

```bash
cd llm-leaderboard
chmod +x scripts/ci/run_zh_remote_pipeline.sh
WANDB_API_KEY=... \
OPENAI_COMPATIBLE_API_KEY=EMPTY \
SUNK_HOST=sunk.cwb607-training.coreweave.app \
SUNK_USER='rsong+cwb607' \
SUNK_SSH_KEY=~/.ssh/id_ed25519_sunk \
scripts/ci/run_zh_remote_pipeline.sh config-foxbrain-zh-tw.yaml
```

## GitHub Actions workflow

Workflow file: `.github/workflows/zh-board-cicd.yml`

Trigger:
- Manual: `workflow_dispatch`
- Scheduled: every Wednesday 02:00 UTC

Required repository secrets:
- `SUNK_HOST`
- `SUNK_USER`
- `SUNK_SSH_PRIVATE_KEY`
- `WANDB_API_KEY`
- `OPENAI_COMPATIBLE_API_KEY` (optional; defaults to `EMPTY` behavior when omitted)

Notes:
- Workflow uses `runs-on: ubuntu-latest` (GitHub-hosted runner).
- Remote Slurm script path defaults to `/mnt/home/<SUNK_USER>/foxbrain_vllm_serve.sh`.
- Override with `REMOTE_VLLM_SCRIPT` env var if needed.
- Pipeline logs are uploaded as GitHub Actions artifacts (`zh-board-cicd-log-*`).
