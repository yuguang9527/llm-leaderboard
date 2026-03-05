# ZH Board Dataset Whitelist Template

Use this template to approve datasets for the Chinese board v1.
One row per dataset source.

## Status Legend

- `GREEN`: approved for commercial use and redistribution (or internal use with clear rights)
- `YELLOW`: usable with constraints; legal review still needed
- `RED`: blocked for current release

## Dataset Inventory

| dataset_id | task_family | subtask | source_url | license | commercial_use | redistribution | attribution_required | copyleft_share_alike | pii_risk | legal_status | risk_level | owner | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tmmluplus | glp.reasoning / glp.knowledge_qa | multi_subject_mcq | https://huggingface.co/datasets/ikala/tmmluplus | MIT (HF api tag) | yes (expected) | yes (expected) | yes | no | low | pending-legal-confirm | YELLOW | data-team | confirm dataset card text and archive screenshot/hash |
| tmlu | glp.reasoning / glp.knowledge_qa | multi_subject_mcq | TBD | TBD | TBD | TBD | TBD | TBD | low | pending | YELLOW | TBD | verify original repo + mirror terms |
| drcd | glp.basic_language | reading_comprehension | TBD | TBD | TBD | TBD | TBD | TBD | low | pending | YELLOW | TBD | confirm share-alike obligations |
| swebench_verified | glp.app_dev | code_repair_exec | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified | UNSPECIFIED on HF card (pending) | unknown | unknown | likely yes | unknown | low | pending | YELLOW | data-team | must verify upstream SWE-bench license and redistribution scope |
| bfcl | glp.app_dev | tool_calling_exec | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | Apache-2.0 | yes | yes | yes | no | low | pending-legal-confirm | YELLOW | data-team | HF README includes apache-2.0; confirm latest upstream repo parity |
| c_safetybench | alt.safety_compliance | toxicity_harm | TBD | TBD | TBD | TBD | TBD | TBD | medium | pending | YELLOW | TBD | verify policy-sensitive content handling requirements |
| cbbq | alt.fairness_bias | bias_qa | https://huggingface.co/datasets/walledai/CBBQ | CC BY-SA 4.0 | yes (with conditions) | yes (share-alike) | yes | yes | low | pending-legal-confirm | YELLOW | data-team | share-alike obligations must be reflected in downstream distribution |
| truthfulqa_zh | alt.truthfulness | truthful_qa | TBD | TBD | TBD | TBD | TBD | TBD | low | pending | YELLOW | TBD | verify derivative translation license |
| halueval_zh | alt.truthfulness | hallucination_resistance | TBD | TBD | TBD | TBD | TBD | TBD | low | pending | YELLOW | TBD | verify benchmark adaptation rights |
| xstest_zh | alt.robustness | jailbreak_resistance | TBD | TBD | TBD | TBD | TBD | TBD | medium | pending | YELLOW | TBD | verify adversarial prompt redistribution rights |
| ifeval_zh | alt.controllability_refusal | instruction_following_refusal | TBD | TBD | TBD | TBD | TBD | TBD | low | pending | YELLOW | TBD | verify translation + redistribution terms |

## Approval Checklist

- [ ] License text captured and archived
- [ ] Commercial use rights confirmed
- [ ] Redistribution rights confirmed
- [ ] Attribution obligations documented
- [ ] Copyleft/share-alike impact reviewed
- [ ] PII/privacy risk reviewed
- [ ] Security policy review completed for harmful content datasets
- [ ] Final status set to GREEN before release

## Artifact Mapping (after approval)

Map approved datasets into artifact versions:

- `zh_board_dataset:v1`
- `zh_board_scoring_spec:v1`
- `zh_board_weights:v1`

