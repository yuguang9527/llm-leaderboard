"""
TMMLU+ evaluator for zh-board-v1.

Evaluates Traditional Chinese multi-subject multiple-choice QA
using the ikala/tmmluplus dataset from HuggingFace.

Scoring: rule-based (exact match on A/B/C/D answer).
"""

import re
from datasets import load_dataset
import pandas as pd
import wandb
import weave

from config_singleton import WandbConfigSingleton


TMMLUPLUS_SUBJECTS = [
    "engineering_math", "dentistry", "traditional_chinese_medicine_clinical_medicine",
    "clinical_psychology", "technical", "culinary_skills", "mechanical",
    "logic_reasoning", "real_estate", "general_principles_of_law",
    "finance_banking", "anti_money_laundering", "ttqav2",
    "marketing_management", "business_management", "organic_chemistry",
    "advance_chemistry", "physics", "secondary_physics", "human_behavior",
    "national_protection", "jce_humanities", "politic_science", "agriculture",
    "official_document_management", "financial_analysis", "pharmacy",
    "educational_psychology", "statistics_and_machine_learning",
    "management_accounting", "introduction_to_law", "computer_science",
    "veterinary_pathology", "accounting", "fire_science", "optometry",
    "insurance_studies", "pharmacology", "taxation", "trust_practice",
    "geography_of_taiwan", "physical_education", "auditing",
    "administrative_law", "education_(profession_level)", "economics",
    "veterinary_pharmacology", "nautical_science",
    "occupational_therapy_for_psychological_disorders",
    "basic_medical_science", "macroeconomics", "trade",
    "chinese_language_and_literature", "tve_design", "junior_science_exam",
    "junior_math_exam", "junior_chinese_exam", "junior_social_studies",
    "tve_mathematics", "tve_chinese_language", "tve_natural_sciences",
    "junior_chemistry", "music", "education", "three_principles_of_people",
    "taiwanese_hokkien",
]

CATEGORY_MAP = {
    "engineering_math": "STEM", "dentistry": "Other", "traditional_chinese_medicine_clinical_medicine": "Other",
    "clinical_psychology": "Other", "technical": "STEM", "culinary_skills": "Other",
    "mechanical": "STEM", "logic_reasoning": "STEM", "real_estate": "Other",
    "general_principles_of_law": "Social Sciences", "finance_banking": "Other",
    "anti_money_laundering": "Other", "ttqav2": "Other",
    "marketing_management": "Social Sciences", "business_management": "Social Sciences",
    "organic_chemistry": "STEM", "advance_chemistry": "STEM", "physics": "STEM",
    "secondary_physics": "STEM", "human_behavior": "Social Sciences",
    "national_protection": "Social Sciences", "jce_humanities": "Humanities",
    "politic_science": "Social Sciences", "agriculture": "Other",
    "official_document_management": "Social Sciences", "financial_analysis": "Other",
    "pharmacy": "Other", "educational_psychology": "Social Sciences",
    "statistics_and_machine_learning": "STEM", "management_accounting": "Other",
    "introduction_to_law": "Social Sciences", "computer_science": "STEM",
    "veterinary_pathology": "Other", "accounting": "Other", "fire_science": "Other",
    "optometry": "Other", "insurance_studies": "Other", "pharmacology": "Other",
    "taxation": "Other", "trust_practice": "Other",
    "geography_of_taiwan": "Social Sciences", "physical_education": "Other",
    "auditing": "Other", "administrative_law": "Social Sciences",
    "education_(profession_level)": "Social Sciences", "economics": "Social Sciences",
    "veterinary_pharmacology": "Other", "nautical_science": "STEM",
    "occupational_therapy_for_psychological_disorders": "Other",
    "basic_medical_science": "Other", "macroeconomics": "Social Sciences",
    "trade": "Social Sciences", "chinese_language_and_literature": "Humanities",
    "tve_design": "Humanities", "junior_science_exam": "STEM",
    "junior_math_exam": "STEM", "junior_chinese_exam": "Humanities",
    "junior_social_studies": "Social Sciences", "tve_mathematics": "STEM",
    "tve_chinese_language": "Humanities", "tve_natural_sciences": "STEM",
    "junior_chemistry": "STEM", "music": "Humanities", "education": "Social Sciences",
    "three_principles_of_people": "Social Sciences", "taiwanese_hokkien": "Humanities",
}


def _extract_answer(text: str) -> str:
    """Extract A/B/C/D from model output."""
    text = text.strip()
    match = re.match(r"^[(\[]?\s*([A-Da-d])\s*[)\]]?", text)
    if match:
        return match.group(1).upper()
    for letter in ["A", "B", "C", "D"]:
        if letter in text.upper():
            return letter
    return text.strip()[:1].upper()


def _format_mcq_prompt(question: str, a: str, b: str, c: str, d: str, few_shot_examples: list | None = None) -> str:
    """Format a multiple-choice prompt in Traditional Chinese."""
    parts = []
    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(
                f"問題：{ex['question']}\n"
                f"(A) {ex['A']}\n(B) {ex['B']}\n(C) {ex['C']}\n(D) {ex['D']}\n"
                f"答案：{ex['answer']}\n"
            )
    parts.append(
        f"問題：{question}\n"
        f"(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
        f"答案："
    )
    return "\n".join(parts)


@weave.op(call_display_name=lambda _: "[zh-TMMLU+] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    bench_cfg = cfg.tmmluplus
    num_few_shots = bench_cfg.get("num_few_shots", 0)
    max_samples_per_subject = bench_cfg.get("max_samples_per_subject", None)
    subjects = bench_cfg.get("subjects", TMMLUPLUS_SUBJECTS)
    system_prompt = bench_cfg.get("system_prompt",
        "你是一位知識淵博的助手。請直接回答選擇題的正確選項字母（A、B、C 或 D），不需要解釋。")

    rows = []
    subject_scores = {}

    for subject in subjects:
        try:
            ds_test = load_dataset("ikala/tmmluplus", subject, split="test")
        except Exception as e:
            print(f"Warning: failed to load subject {subject}: {e}")
            continue

        few_shot_examples = []
        if num_few_shots > 0:
            try:
                ds_dev = load_dataset("ikala/tmmluplus", subject, split="train")
                few_shot_examples = [ds_dev[i] for i in range(min(num_few_shots, len(ds_dev)))]
            except Exception:
                pass

        samples = list(ds_test)
        if max_samples_per_subject and len(samples) > max_samples_per_subject:
            samples = samples[:max_samples_per_subject]

        correct = 0
        total = 0

        for idx, sample in enumerate(samples):
            prompt_text = _format_mcq_prompt(
                sample["question"], sample["A"], sample["B"], sample["C"], sample["D"],
                few_shot_examples if num_few_shots > 0 else None
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]
            response = llm.invoke(messages=messages, max_tokens=int(bench_cfg.get("max_tokens", 32)))
            raw_output = response.content if hasattr(response, "content") else str(response)
            predicted = _extract_answer(raw_output)
            gold = sample["answer"].strip().upper()
            is_correct = predicted == gold
            if is_correct:
                correct += 1
            total += 1

            rows.append({
                "subject": subject,
                "category": CATEGORY_MAP.get(subject, "Other"),
                "index": idx,
                "question": sample["question"][:200],
                "gold": gold,
                "predicted": predicted,
                "raw_output": raw_output[:500],
                "correct": int(is_correct),
            })

        if total > 0:
            subject_scores[subject] = correct / total

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError("TMMLU+ evaluation produced no rows")

    category_scores = output_df.groupby("category", as_index=False)["correct"].mean()
    category_scores.rename(columns={"correct": "accuracy"}, inplace=True)

    leaderboard = {"model_name": cfg.model.pretrained_model_name_or_path}
    for _, row in category_scores.iterrows():
        leaderboard[row["category"]] = float(row["accuracy"])
    leaderboard["AVG"] = float(output_df["correct"].mean())
    leaderboard_df = pd.DataFrame([leaderboard])

    run.log({
        "tmmluplus_output_table": wandb.Table(dataframe=output_df),
        "tmmluplus_leaderboard_table": wandb.Table(dataframe=leaderboard_df),
    })
