"""
バッチ推論後にWeaveにログする軽量ラッパー
"""

from typing import Any, Dict, List, Optional
import logging

try:
    from weave import EvaluationLogger
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    EvaluationLogger = None

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
        name: Optional[str] = None,
        eval_attributes: Optional[Dict[str, Any]] = None,
        multi_turn: bool = False,
    ):
        """
        Args:
            dataset_name: データセット名（例: "jaster_0shot", "jaster_2shot"）
            model_name: モデル名（例: "gpt-4o-2024-11-20"）
            name: 評価の名前（ベンチマーク名、例: "jaster"）
            eval_attributes: 追加の評価属性（サブセット、スプリットなど）
            multi_turn: マルチターン形式かどうか（Trueの場合messagesを出力に含める）
        """
        self.dataset_name = dataset_name
        self.name = name
        # Weaveはモデル名にハイフン等を許容しないため正規化
        self.model_label = model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        self.eval_attributes = eval_attributes or {}
        self.multi_turn = multi_turn
        self.elog: Optional["EvaluationLogger"] = None
        self._enabled = WEAVE_AVAILABLE

    def initialize(self) -> "WeaveEvalLogger":
        """EvaluationLoggerを初期化"""
        if not self._enabled:
            logger.warning("Weave not available, skipping initialization")
            return self

        self.elog = EvaluationLogger(
            name=self.name,
            dataset=self.dataset_name,
            model=self.model_label,
        )
        logger.info(f"Weave EvaluationLogger initialized: name={self.name}, dataset={self.dataset_name}, model={self.model_label}")
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
        if not self._enabled or self.elog is None:
            logger.warning("Weave not initialized, skipping log_samples")
            return

        logger.info(f"Logging {len(samples)} samples to Weave")
        for sample in samples:
            inputs_payload, output_payload = self._build_payloads(sample)

            # 予測をログ
            pred_logger = self.elog.log_prediction(
                inputs=inputs_payload,
                output=output_payload,
            )

            # スコアをログ（final_score のみ除く）
            eval_scores = sample.get(scores_key, {})
            for scorer_name, score_value in eval_scores.items():
                if (isinstance(score_value, (int, float)) and
                    scorer_name not in ["final_score"]):
                    pred_logger.log_score(scorer=str(scorer_name), score=score_value)

            # 予測ログを完了
            pred_logger.finish()

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
            
            if self.multi_turn:
                # マルチターンの場合は全会話を出力に含める
                output_payload = {
                    "messages": messages,  # 全会話
                    "content": prediction,  # 最終出力を別フィールドにも
                }
            else:
                # シングルターンの場合はcontentのみ
                output_payload = {
                    "content": prediction,
                }
        else:
            # フォールバック：単純なinput/output形式
            inputs_payload = {"input": sample.get("input", "")}
            output_payload = prediction
        
        # 参照情報があれば追加
        if reference is not None:
            inputs_payload["reference"] = reference
        
        # メタデータを追加
        for key in ["id", "subset", "_subset_name", "task", "index"]:
            if key in sample and sample[key] is not None:
                inputs_payload[key] = str(sample[key])
        
        return inputs_payload, output_payload

    def finalize(self, summary_metrics: Optional[Dict[str, Any]] = None) -> None:
        """サマリーをログして終了"""
        if not self._enabled or self.elog is None:
            return
        
        self.elog.log_summary(summary=summary_metrics or {})
        self.elog.finish()
        logger.info(f"Weave logging完了: {self.dataset_name}")


