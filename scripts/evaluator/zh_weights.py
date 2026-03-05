"""
Parameterized weight engine for zh-board-v1.

Reads weights from zh_board_v1_spec.yaml and computes
GLP_AVG, ALT_AVG, TOTAL_SCORE using configurable weights
instead of hardcoded averages.
"""

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np


_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "zh_board_v1_spec.yaml"

_FIELD_MAP = {
    "glp": {
        "basic_language": "GLP_基礎語言",
        "reasoning": "GLP_繁中知識推理(TMMLU+)",
        "knowledge_qa": "GLP_知識問答",
        "app_dev": "GLP_應用開發",
    },
    "alt": {
        "safety_compliance": "ALT_安全合規",
        "fairness_bias": "ALT_偏見公平(CBBQ)",
        "truthfulness": "ALT_真實性",
        "robustness": "ALT_魯棒性",
        "controllability_refusal": "ALT_可控拒答",
    },
}


def load_weights(spec_path: str | Path | None = None) -> dict:
    path = Path(spec_path) if spec_path else _SPEC_PATH
    if not path.exists():
        raise FileNotFoundError(f"ZH board spec not found: {path}")
    spec = OmegaConf.load(path)
    return OmegaConf.to_container(spec.weights, resolve=True)


def compute_zh_scores(leaderboard_dict: dict, weights: dict | None = None) -> dict:
    """Compute GLP_AVG, ALT_AVG, TOTAL_SCORE from leaderboard_dict using weights."""
    if weights is None:
        weights = load_weights()

    glp_weights = weights.get("glp", {})
    alt_weights = weights.get("alt", {})
    total_weights = weights.get("total_score", {"glp": 0.6, "alt": 0.4})

    def _weighted_avg(category_weights: dict, field_map: dict) -> float:
        scores = []
        w = []
        for key, weight in category_weights.items():
            field_name = field_map.get(key)
            if field_name and field_name in leaderboard_dict:
                val = leaderboard_dict[field_name]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    scores.append(val)
                    w.append(weight)
        if not scores:
            return float('nan')
        total_w = sum(w)
        return sum(s * wi / total_w for s, wi in zip(scores, w))

    glp_avg = _weighted_avg(glp_weights, _FIELD_MAP["glp"])
    alt_avg = _weighted_avg(alt_weights, _FIELD_MAP["alt"])

    leaderboard_dict["汎用的言語性能(GLP)_AVG"] = glp_avg
    leaderboard_dict["アラインメント(ALT)_AVG"] = alt_avg

    glp_w = total_weights.get("glp", 0.6)
    alt_w = total_weights.get("alt", 0.4)

    available = []
    if not np.isnan(glp_avg):
        available.append((glp_avg, glp_w))
    if not np.isnan(alt_avg):
        available.append((alt_avg, alt_w))

    if available:
        total_w = sum(w for _, w in available)
        leaderboard_dict["TOTAL_SCORE"] = sum(s * w / total_w for s, w in available)
    else:
        leaderboard_dict["TOTAL_SCORE"] = float('nan')

    return leaderboard_dict
