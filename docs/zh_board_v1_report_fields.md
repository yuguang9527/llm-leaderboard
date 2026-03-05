# ZH Board V1 - W&B Report Field Reference

W&B Report queries use `runs.summary["leaderboard_table"]`.
Below is the complete field list for the Chinese board v1.

## Core Fields (always present)

| Field | Type | Source |
|---|---|---|
| `model_name` | string | config |
| `model_size_category` | string | config |
| `model_size` | number | config |
| `model_release_date` | date | config |
| `base_model` | string | config |
| `TOTAL_SCORE` | float | zh_weights engine |

## GLP Fields (General Language Performance)

| Field | Type | Source Benchmark | Scoring |
|---|---|---|---|
| `汎用的言語性能(GLP)_AVG` | float | weighted avg of GLP sub-items | zh_weights engine |
| `GLP_繁中知識推理(TMMLU+)` | float | tmmluplus | rule (exact match MCQ) |
| `GLP_基礎語言` | float | (future: DRCD etc.) | rule |
| `GLP_知識問答` | float | (future: TMMLU+ subset mapping) | rule |
| `GLP_應用開發` | float | SWE-bench + BFCL | exec |
| `GLP_繁體中文` | float | taiwan_zh_tw (legacy PoC) | rule |
| `GLP_コーディング` | float | SWE-bench + BFCL coding | exec |
| `GLP_関数呼び出し` | float | BFCL | exec |

## ALT Fields (Alignment)

| Field | Type | Source Benchmark | Scoring |
|---|---|---|---|
| `アラインメント(ALT)_AVG` | float | weighted avg of ALT sub-items | zh_weights engine |
| `ALT_偏見公平(CBBQ)` | float | cbbq_zh | rule (exact match) |
| `ALT_安全合規` | float | (future: C-SafetyBench) | judge |
| `ALT_真實性` | float | (future: TruthfulQA-zh + HaluEval-zh) | judge |
| `ALT_魯棒性` | float | (future: XSTest-zh) | rule |
| `ALT_可控拒答` | float | (future: IFEval-zh) | rule |

## Per-Benchmark Detail Tables

These are logged as separate `wandb.Table` objects (not in `leaderboard_table`):

| Table Name | Content |
|---|---|
| `tmmluplus_output_table` | Per-question results (subject, question, gold, predicted, correct) |
| `tmmluplus_leaderboard_table` | Per-category accuracy + AVG |
| `cbbq_zh_output_table` | Per-question bias results |
| `cbbq_zh_leaderboard_table` | Per-category accuracy + bias score |
| `taiwan_zh_tw_output_table` | Legacy PoC per-question results |
| `taiwan_zh_tw_leaderboard_table` | Legacy PoC per-task + AVG |
| `subcategory_table_tmmluplus` | TMMLU+ category breakdown |
| `subcategory_table_cbbq_zh` | CBBQ category breakdown |
| `subcategory_table_taiwan_zh_tw` | Legacy PoC breakdown |
| `swebench_leaderboard_table` | SWE-bench resolution rate (from existing Nejumi evaluator) |
| `bfcl_leaderboard_table` | BFCL overall/category accuracy (from existing Nejumi evaluator) |

## Weight Configuration

Weights are read from `configs/zh_board_v1_spec.yaml`:

```yaml
weights:
  total_score:
    glp: 0.6
    alt: 0.4
  glp:
    basic_language: 0.15
    reasoning: 0.20
    knowledge_qa: 0.15
    app_dev: 0.10
  alt:
    safety_compliance: 0.12
    fairness_bias: 0.08
    truthfulness: 0.08
    robustness: 0.06
    controllability_refusal: 0.06
```

## Report Query Examples

Main leaderboard table:
```
runs.summary["leaderboard_table"]
```

Filter by model size:
```
runs.summary["leaderboard_table"].filter(model_size_category == "Large (30B+)")
```

Sort by total score:
```
runs.summary["leaderboard_table"].sort(TOTAL_SCORE, desc)
```
