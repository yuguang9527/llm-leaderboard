# Goal 
- LLM-leaderboardをW&B Modelsを元に作っているが、Weaveとの連携も入れたい
- まずは、mtbenchとjasterだけで良い

# 基本方針
- 進捗はlocal gitで管理しつつ、後でもどれるように丁寧に進めて

# test command
-  
- または直接docker execで実行:
  ```bash
  docker exec llm-leaderboard bash -c "cd /workspace && source .venv/bin/activate && python -u scripts/run_eval.py --config config-gpt-4.1-nano-2025-04-14.yaml"
  ```
（テストコードがうまくいかなかったら原因を特定しつつ、updateしていって）
- うまくいっているかどうかは、人間が判断するから、勝手にテストコードは作らないで。また、勝手に性能が変わるような変更は入れないで、あくまでTraceをしたい


# 現在の構成
## 全体アーキテクチャ
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         scripts/run_eval.py                             │
│                           (エントリーポイント)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   configs/    │         │ W&B Integration │         │    Evaluators   │
│ (YAML設定)    │         │                 │         │  (ベンチマーク)  │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                         │                           │
        ▼                         ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│config_singleton│        │ wandb.init()    │         │  LLM Inference  │
│ (設定管理)     │         │ weave.init()    │         │    Adapter      │
└───────────────┘         └─────────────────┘         └─────────────────┘
```

## 1. エントリーポイント (`scripts/run_eval.py`)

評価パイプラインの起点。以下の処理フローを実行：

1. **設定読み込み**: `configs/` から YAML ファイルを読み込み、`base_config.yaml` とマージ
2. **W&B 初期化**: `wandb.init()` で実験トラッキング開始、`weave.init()` でトレーシング開始
3. **LLM エンジン取得**: `get_llm_inference_engine()` で推論クライアントを初期化
4. **ベンチマーク実行**: 設定で有効化されたベンチマークを順次実行
5. **結果集計**: `aggregate.evaluate()` でスコアを集約し W&B にログ

## 2. 設定システム (`configs/`)

### 構造
```
configs/
├── base_config.yaml          # ベース設定（全モデル共通）
├── config-{model-name}.yaml  # モデル固有の設定
└── ...
```

### 設定階層
- **wandb**: entity, project, run_name
- **run**: 実行するベンチマークのフラグ (bfcl, swebench, mtbench, jbbq, etc.)
- **model**: モデル名、アーティファクトパス
- **generator**: LLM 生成パラメータ (temperature, max_tokens, top_p)
- **vllm**: vLLM サーバー設定
- **{benchmark}**: 各ベンチマーク固有の設定

### 設定管理 (`config_singleton.py`)
```python
class WandbConfigSingleton:
    # シングルトンパターンで設定をグローバル管理
    # run, config, blend_config, llm を保持
```

## 3. LLM 推論アダプター (`scripts/llm_inference_adapter.py`)

複数の LLM プロバイダーを統一インターフェースで抽象化。

### サポート API タイプ
| API タイプ | クライアントクラス | 説明 |
|-----------|------------------|------|
| `openai`, `openai_chat` | `OpenAIClient` | OpenAI Chat Completions API |
| `openai_responses` | `OpenAIResponsesClient` | OpenAI Responses API (Reasoning対応) |
| `vllm`, `vllm-docker` | `OpenAIClient` | vLLM OpenAI互換API |
| `openai-compatible` | `OpenAIClient` | 汎用 OpenAI 互換 API |
| `anthropic` | `AnthropicClient` | Anthropic Claude API |
| `google` | `GoogleClient` | Google Gemini API |
| `mistral` | `MistralClient` | Mistral AI API |
| `cohere` | `CohereClient` | Cohere API (v1/v2対応) |
| `amazon_bedrock` | `ChatBedrock` | AWS Bedrock (Claude, Llama, Nova) |
| `azure-openai` | `AzureOpenAIClient` | Azure OpenAI Service |
| `xai` | `OpenAIClient` | xAI Grok API |
| `upstage` | `OpenAIClient` | Upstage Solar API |

### 共通インターフェース
```python
class BaseLLMClient(ABC):
    def invoke(messages, max_tokens, **kwargs) -> LLMResponse  # 同期
    async def ainvoke(messages, max_tokens, **kwargs) -> LLMResponse  # 非同期

@dataclass
class LLMResponse:
    content: str
    reasoning_content: str = ""  # Reasoning モデル対応
    parsed_output: Optional[BaseModel] = None  # Structured Output
    tool_calls: Optional[List[ToolCall]] = None  # Function Calling
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
```

## 4. ベンチマーク評価器 (`scripts/evaluator/`)

### 実装ベンチマーク一覧

| モジュール | ベンチマーク | 説明 |
|-----------|------------|------|
| `bfcl.py` | BFCL | Berkeley Function Calling Leaderboard |
| `swe_bench.py` | SWE-Bench | ソフトウェアエンジニアリングベンチマーク |
| `mtbench.py` | MT-Bench | マルチターン対話評価 |
| `jbbq.py` | JBBQ | 日本語バイアス評価 |
| `toxicity.py` | Toxicity | 毒性評価 |
| `jtruthfulqa.py` | JTruthfulQA | 日本語真実性評価 |
| `hle.py` | HLE | Humanity's Last Exam |
| `hallulens.py` | HalluLens | ハルシネーション評価 |
| `arc_agi.py` | ARC-AGI | 抽象推論コーパス |
| `m_ifeval.py` | M-IFEval | 多言語指示追従評価 |
| `jaster.py` | Jaster | 日本語タスク統合評価 |
| `jaster_translation.py` | Translation | 翻訳品質評価 (COMET) |
| `aggregate.py` | Aggregate | スコア集約・最終評価 |

### 評価ユーティリティ (`scripts/evaluator/evaluate_utils/`)
- `llm_async_processor.py`: 非同期バッチ処理
- `llm_judge_client.py`: LLM-as-a-Judge クライアント
- `progress_tracker.py`: 進捗トラッキング
- `validation_helpers.py`: トークン割り当てバリデーション
- `metrics.py`: メトリクス計算
- `bfcl_pkg/`: BFCL 評価パッケージ
- `swebench_pkg/`: SWE-Bench 評価パッケージ

## 5. W&B 連携

### 現在の実装
```python
# run_eval.py L107-117
wandb.login()
run = wandb.init(
    entity=cfg_dict["wandb"]["entity"],
    project=cfg_dict["wandb"]["project"],
    name=cfg_dict["wandb"]["run_name"],
    config=cfg_dict,
    job_type="evaluation",
)
weave.init(cfg_dict["wandb"]["entity"]+"/"+cfg_dict["wandb"]["project"])
```

### W&B Models 機能
- **実験トラッキング**: `wandb.run` で評価メトリクスをログ
- **Artifacts**: データセット、設定、結果の保存
- **Run チェイン**: `blend_run()` で過去の Run を継承

### Weave 連携 (現状)
- `weave.init()` でプロジェクト初期化のみ
- **拡張の余地**: LLM 呼び出しのトレース、コスト追跡、デバッグ

### Weave EvalLogger 統合

#### 概要
`WeaveEvalLogger` を使用して、各ベンチマークの評価結果をWeaveにログできます。

#### 基本的な使用方法

1. **WeaveEvalLogger の初期化**
```python
from scripts.evaluator.evaluate_utils import WeaveEvalLogger

weave_logger = WeaveEvalLogger(
    dataset_name=f"{benchmark_name}_{few_shots}shot",  # 例: "jaster_0shot"
    model_name=cfg.wandb.run_name,  # W&B run nameを使用
    name=benchmark_name,  # ベンチマーク名 (例: "jaster")
    eval_attributes={
        "num_few_shots": few_shots,  # 追加属性
        "multi_turn": False,  # マルチターンかどうか
    },
)
weave_logger.initialize()
```

2. **サンプルデータの準備**
```python
# 各ベンチマークの評価結果をWeave形式に変換
sample_key_to_data = {}
for er in evaluation_results:
    key = (er["task"], er["subset"], er["index"])
    if key not in sample_key_to_data:
        sample_key_to_data[key] = {
            "messages": er.get("messages_for_log", []),  # ChatGPT形式の会話履歴
            "prediction": er["output"],  # モデルの出力
            "reference": er["expected_output"],  # 正解
            "input": er["input"],  # 入力
            "task": er["task"],  # ベンチマーク名
            "subset": er["subset"],  # subset (test/devなど)
            "index": er["index"],  # サンプルインデックス
            "evaluation": {},  # スコア格納用
        }

    # 各メトリクスのスコアを追加
    metrics_name = er["metrics"]
    if er["score"] is not None and not np.isnan(er["score"]):
        sample_key_to_data[key]["evaluation"][metrics_name] = er["score"]

    # primary_metric の場合は primary_score としても追加
    if er["metrics"] == er["primary_metric"]:
        sample_key_to_data[key]["evaluation"]["primary_score"] = er["score"]
```

3. **Weaveへのログ**
```python
# 全サンプルをログ
weave_logger.log_samples(list(sample_key_to_data.values()))
```

4. **Summaryのログ**
```python
# 全subsetの各ベンチマークの primary_score 平均を集計
summary_metrics = {}
for sample in sample_key_to_data.values():
    task = sample["task"]
    final_score = sample.get("evaluation", {}).get("primary_score")

    if final_score is not None and isinstance(final_score, (int, float)):
        if task not in summary_metrics:
            summary_metrics[task] = []
        summary_metrics[task].append(final_score)

avg_summary = {k: np.mean(v) for k, v in summary_metrics.items() if v}
weave_logger.finalize(summary_metrics=avg_summary)
```

#### ベンチマークごとの統合方法

各ベンチマークの`evaluate()`関数内で、以下の条件分岐を追加：

```python
# Weave EvalLogger integration
if cfg.get("weave_evallogger_integration", False):
    # 上記のサンプルデータ準備とログ処理
    weave_logger = WeaveEvalLogger(...)
    weave_logger.initialize()
    weave_logger.log_samples(filtered_samples)
    weave_logger.finalize(summary_metrics=avg_summary)
```

#### 設定方法

`base_config.yaml`または各モデルのconfigファイルに以下を追加：

```yaml
weave_evallogger_integration: true  # Weaveログを有効化
```

#### ログされるデータ構造

- **各サンプル**: `messages`, `prediction`, `reference`, `input`, `task`, `subset`, `index`, `evaluation`
- **Evaluationスコア**: `exact_match`, `char_f1`, `primary_score` など（char_f1は除外可能）
- **Summary**: 各ベンチマークの `primary_score` 平均

#### マルチターン対応

マルチターンベンチマークの場合は、`eval_attributes`に `multi_turn: true` を設定し、出力ペイロードに全会話の `messages` を含めます。

#### 注意点

- `final_score` はログ対象外（`primary_score` を使用）
- `char_f1` などの補助スコアは必要に応じて除外可能
- `test` subsetのみ、または全subsetを集計可能
- ベンチマーク名は `dataset_name` と `name` で分離（例: dataset="jaster_0shot", name="jaster"）

## 6. 実行フロー

```
1. CLI引数解析 → 設定ファイル選択
2. base_config.yaml + モデル設定 をマージ
3. W&B/Weave 初期化
4. WandbConfigSingleton に設定を保存
5. トークンバリデーション実行
6. プログレストラッカー初期化
7. LLM推論エンジン取得 (get_llm_inference_engine)
8. 各ベンチマーク実行:
   - BFCL → SWE-Bench → MT-Bench → JBBQ → Toxicity
   → JTruthfulQA → HLE → HalluLens → ARC-AGI → M-IFEval → Jaster
9. 結果集計 (aggregate)
10. W&B Run 終了
```

## 7. ディレクトリ構造

```
llm-leaderboard/
├── scripts/
│   ├── run_eval.py              # エントリーポイント
│   ├── config_singleton.py      # 設定シングルトン
│   ├── llm_inference_adapter.py # LLM クライアント抽象化
│   ├── blend_run.py             # Run継承
│   ├── vllm_server.py           # vLLM サーバー管理
│   ├── docker_vllm_manager.py   # Docker vLLM 管理
│   └── evaluator/               # ベンチマーク評価器
│       ├── __init__.py
│       ├── {benchmark}.py       # 各ベンチマーク実装
│       └── evaluate_utils/      # 共通ユーティリティ
├── configs/                     # 設定ファイル
├── artifacts/                   # ローカルアーティファクト
├── chat_templates/              # カスタムチャットテンプレート
└── src/                         # サードパーティソース
    ├── fschat/                  # FastChat
    └── llm-kr-eval/             # 韓国語評価
```


## 目指していきたいWeaveでのlogging
- Evaluation Loggerを使ってinputとoutputを保存
- eval_logger = EvaluationLogger(
    model="my_model", <- ここで実装したモデルnameを使う. confingのrun nameをとってくる
    dataset="my_dataset" <- ここで実装するbenchmark名を入れる
)
- 以下はあくまで例だが、Weaveでlogするclassを作り、それに対して既存benchmarkのinputとoutputを当てはめるようにしていきたい


"""
バッチ推論後にWeaveにログする軽量ラッパー
"""

from typing import Any, Dict, List, Optional
from weave import EvaluationLogger
import logging

logger = logging.getLogger(__name__)


class WeaveEvalLogger:
    """
    バッチ推論後にWeaveへログするシンプルなラッパー。
    ChatCompletion形式で会話履歴を保存する。
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        eval_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        # Weaveはモデル名にハイフン等を許容しないため正規化
        self.model_label = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        self.eval_attributes = eval_attributes or {}
        self.elog: Optional[EvaluationLogger] = None

    def initialize(self) -> "WeaveEvalLogger":
        """EvaluationLoggerを初期化"""
        self.elog = EvaluationLogger(
            dataset=self.dataset_name,
            model=self.model_label,
            eval_attributes=self.eval_attributes,
        )
        return self

    def log_samples(
        self,
        samples: List[Dict[str, Any]],
        scores_key: str = "evaluation",
    ) -> None:
        """
        バッチ推論済みサンプルをWeaveにログ。
        
        各サンプルに以下のキーを期待：
        - messages: List[Dict] - ChatCompletion形式の会話履歴
        - prediction: str - モデルの出力（messagesがない場合のフォールバック）
        - reference: str (optional) - 正解
        - evaluation: Dict[str, float] (optional) - 評価スコア
        """
        if self.elog is None:
            raise RuntimeError("initialize()を先に呼んでください")

        for sample in samples:
            inputs_payload, output_payload = self._build_payloads(sample)
            
            # 予測をログ
            pred_logger = self.elog.log_prediction(
                inputs=inputs_payload,
                output=output_payload,
            )
            
            # スコアをログ
            eval_scores = sample.get(scores_key, {})
            for scorer_name, score_value in eval_scores.items():
                if isinstance(score_value, (int, float)):
                    pred_logger.log_score(scorer=str(scorer_name), score=score_value)

    def _build_payloads(self, sample: Dict[str, Any]) -> tuple:
        """
        ChatCompletion形式のペイロードを構築。
        Weave UIで会話履歴として表示される。
        """
        messages = sample.get("messages", [])
        prediction = sample.get("prediction", "")
        reference = sample.get("reference")
        
        if messages:
            # ChatCompletion形式：会話履歴として表示
            # 最後のassistantメッセージがない場合は追加
            if not any(m.get("role") == "assistant" for m in messages):
                messages = messages + [{"role": "assistant", "content": prediction}]
            
            inputs_payload = {
                "messages": messages[:-1] if messages else [],  # assistant以外
            }
            output_payload = {
                "messages": messages,  # 全会話
                "content": prediction,  # 最終出力を別フィールドにも
            }
        else:
            # フォールバック：単純なinput/output形式
            inputs_payload = {"input": sample.get("input", "")}
            output_payload = prediction
        
        # 参照情報があれば追加
        if reference is not None:
            inputs_payload["reference"] = reference
        
        # メタデータを追加
        for key in ["id", "subset", "_subset_name"]:
            if key in sample and sample[key] is not None:
                inputs_payload[key] = str(sample[key])
        
        return inputs_payload, output_payload

    def finalize(self, summary_metrics: Optional[Dict[str, Any]] = None) -> None:
        """サマリーをログして終了"""
        if self.elog is None:
            return
        
        self.elog.log_summary(summary=summary_metrics or {})
        self.elog.finish()
        logger.info(f"Weave logging完了: {self.dataset_name}")
- 上記の実装例。あくまで例
import weave
from weave_logger import WeaveEvalLogger

# Weave初期化
weave.init("your-entity/your-project")

# 1. バッチ推論を実行（既存のコードを使用）
samples = [
    {"input": "韓国の首都は？", "reference": "ソウル"},
    {"input": "日本の首都は？", "reference": "東京"},
    ...
]
results = model.generate_batch(samples)  # ← バッチ処理OK！

# 2. 評価を実行
for sample in results:
    sample["evaluation"] = evaluator.score(sample)

# 3. Weaveにログ
logger = WeaveEvalLogger(
    dataset_name="kmmlu",
    model_name="gpt-4o-2024-11-20",
    eval_attributes={
        "subset": ["accounting", "biology"],
        "split": "test",
    },
)
logger.initialize()
logger.log_samples(results)
logger.finalize(summary_metrics={"accuracy": 0.85})


- multi-turnのbenchmarkには印がつけておく必要がある <- base_config.yamlの各benchmarkの中にmulti_turn: true, falseみたいな感じで追加して、multi-turnだったら、outputをmessagesで入れるようにして
- /home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluatorの中に各benchmarkの実装がある基本evaluate関数があるはず。そして最後にWandbのTableなどを保存するところがあるが、base_configでweave_evallogger_integrationがtrueだったら、evalloggerでloggingをしたい




