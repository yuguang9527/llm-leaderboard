import json
import re
from pathlib import Path

import pandas as pd
import wandb
import weave

from config_singleton import WandbConfigSingleton


def _normalize(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    # Normalize whitespace and common variant characters for zh-TW matching.
    text = re.sub(r"\s+", " ", text)
    text = text.replace("臺", "台")
    return text


def _score_prediction(metric: str, prediction: str, answers: list[str]) -> float:
    prediction_n = _normalize(prediction)
    answers_n = [_normalize(a) for a in answers]

    if metric == "exact_any":
        return 1.0 if any(prediction_n == ans for ans in answers_n) else 0.0
    if metric == "contains_any":
        return 1.0 if any(ans in prediction_n for ans in answers_n) else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


@weave.op(call_display_name=lambda _: "[zh-TW] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    bench_cfg = cfg.taiwan_zh_tw
    dataset_path = Path(bench_cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"taiwan_zh_tw dataset not found: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", [])

    max_samples = bench_cfg.max_samples_testmode if cfg.testmode else bench_cfg.max_samples
    selected_samples = samples[:max_samples]

    rows = []
    for idx, sample in enumerate(selected_samples):
        messages = [
            {"role": "system", "content": bench_cfg.system_prompt},
            {"role": "user", "content": sample["prompt"]},
        ]
        response = llm.invoke(messages=messages, max_tokens=int(bench_cfg.max_tokens))
        raw_output = response.content if hasattr(response, "content") else str(response)
        metric = sample.get("metric", "contains_any")
        score = _score_prediction(metric=metric, prediction=raw_output, answers=sample["answers"])

        rows.append(
            {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "dataset": "taiwan_zh_tw",
                "task": sample["task"],
                "index": idx,
                "prompt": sample["prompt"],
                "raw_output": raw_output,
                "answers": " | ".join(sample["answers"]),
                "metric": metric,
                "score": score,
            }
        )

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError("taiwan_zh_tw evaluation produced no rows")

    task_scores = output_df.groupby("task", as_index=False)["score"].mean()
    leaderboard = {"model_name": cfg.model.pretrained_model_name_or_path}
    for _, row in task_scores.iterrows():
        leaderboard[row["task"]] = float(row["score"])
    leaderboard["AVG"] = float(output_df["score"].mean())
    leaderboard_df = pd.DataFrame([leaderboard])

    run.log(
        {
            "taiwan_zh_tw_output_table": wandb.Table(dataframe=output_df),
            "taiwan_zh_tw_leaderboard_table": wandb.Table(dataframe=leaderboard_df),
        }
    )
