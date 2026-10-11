Canonical LLM results selection for the paper.

This folder is the single source of truth for experiment outputs kept in the repo.

Selection policy:
- Keep cross-dataset transformer results (`cross_dataset_seed*`, `model_c_deberta_focal_mask`) unchanged.
- Keep post-fix LLM reruns where available:
  - `model_d1_llm_zeroshot_seed42-44`
  - `model_d2_llm_fewshot_seed42-44`
  - `model_d3_llm_guided_cot_seed42-44`
  - `model_d2_mistral_fewshot_seed42-44`
  - `model_d3_mistral_guided_cot_seed42-44`
  - `model_d2_qwen_fewshot_seed42-44`
  - `model_d3_qwen_guided_cot_seed42-44`
- Keep legacy single-run zero-shot baselines only where no post-fix rerun exists:
  - `model_d1_mistral_zeroshot`
  - `model_d1_qwen_zeroshot`

Excluded from the canonical set:
- Smoke runs
- Older duplicated LLM runs superseded by the post-fix reruns
- Mixed old/new duplicates for the same model/prompt/seed
