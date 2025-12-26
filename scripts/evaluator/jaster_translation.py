import pandas as pd
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton
from .evaluate_utils import commet_score, WeaveEvalLogger


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    num_few_shots=cfg.num_few_shots
    dataset_name = "jaster"
    for i in [0,num_few_shots]:
        jaster_output_table_dev=read_wandb_table(f"{dataset_name}_{i}shot_output_table_dev", run=run)
        jaster_output_table_test=read_wandb_table(f"{dataset_name}_{i}shot_output_table", run=run)
        
        updated_output_table_dev = pd.DataFrame(add_comet_evaluation_result(jaster_output_table_dev.to_dict(orient='records')))
        updated_output_table_test = pd.DataFrame(add_comet_evaluation_result(jaster_output_table_test.to_dict(orient='records')))

        leaderboard_table = pd.pivot_table(
            data=updated_output_table_test,
            values="score",
            index="model_name",
            columns="task",
            aggfunc="mean",
        ).reset_index()

        leaderboard_table.drop(columns=["model_name"], inplace=True)
        leaderboard_table.insert(0, 'AVG', leaderboard_table.iloc[:, 2:].mean(axis=1))
        leaderboard_table.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
    
        run.log(
            {
                f"{dataset_name}_{i}shot_output_table_dev": updated_output_table_dev,
                f"{dataset_name}_{i}shot_output_table": updated_output_table_test,
                f"{dataset_name}_{i}shot_leaderboard_table": leaderboard_table,
            }
        )

        # Weave EvalLogger integration (after COMET score calculation)
        if cfg.get("weave_evallogger_integration", False):
            print(f"[INFO] Weave logging for {dataset_name}_{i}shot")

            # Prepare sample data for Weave (only test subset)
            sample_key_to_data = {}
            test_records = updated_output_table_test.to_dict('records')

            for record in test_records:
                key = (record["task"], record["subset"], record["index"])
                if key not in sample_key_to_data:
                    # Reconstruct messages for logging
                    messages = []
                    if record.get("prompt"):
                        # Simple reconstruction - user message only
                        messages = [{"role": "user", "content": record["input"]}]

                sample_key_to_data[key] = {
                    "messages": messages,
                    "prediction": record["output"],
                    "reference": record["expected_output"],
                    "input": record["input"],
                    "task": record["task"],
                    "subset": record["subset"],
                    "index": record["index"],
                    "num_few_shots": i,  # Add num_few_shots to sample data
                    "evaluation": {},
                }

                # Add evaluation scores
                if record["score"] is not None and not (isinstance(record["score"], float) and pd.isna(record["score"])):
                    metrics_name = record["metrics"]
                    sample_key_to_data[key]["evaluation"][metrics_name] = record["score"]

                    # Add primary_score (first metric of each task)
                    if record["metrics"] == record.get("primary_metric", record["metrics"]):
                        sample_key_to_data[key]["evaluation"]["primary_score"] = record["score"]

            # WeaveEvalLogger for final scores
            weave_logger = WeaveEvalLogger(
                dataset_name=f"{dataset_name}_{i}shot",
                model_name=cfg.wandb.run_name,
                name=dataset_name,
                eval_attributes={
                    "num_few_shots": i,
                },
                multi_turn=False,
            )
            weave_logger.initialize()
            weave_logger.log_samples(list(sample_key_to_data.values()))

            # Summary metrics (task-wise primary_score averages)
            summary_metrics = {}
            for sample in sample_key_to_data.values():
                task = sample["task"]
                final_score = sample.get("evaluation", {}).get("primary_score")

                if final_score is not None and isinstance(final_score, (int, float)):
                    if task not in summary_metrics:
                        summary_metrics[task] = []
                    summary_metrics[task].append(final_score)

            avg_summary = {k: pd.Series(v).mean() for k, v in summary_metrics.items() if v}
            weave_logger.finalize(summary_metrics=avg_summary)

def add_comet_evaluation_result(evaluation_results):
    # インデックスとスコアを計算するためのリストを初期化
    indices = []
    commet_src = []
    commet_mt = []
    commet_ref = []

    # evaluation_resultsの各要素（辞書）をループ処理
    for i, result in enumerate(evaluation_results):
        if "comet_wmt22" in result["metrics"]:
            # 'comet_wmt22' を持つインデックスをリストに追加
            indices.append(i)
            commet_src.append(result["input"])
            commet_mt.append(result["output"])
            commet_ref.append(result["expected_output"])

    if indices:  # インデックスリストが空でない場合
        # comet_scoreを計算
        commet_scores = commet_score(commet_src, commet_mt, commet_ref)

        # 結果を更新
        for index, new_score in zip(indices, commet_scores):
            evaluation_results[index]["score"] = new_score

    return evaluation_results