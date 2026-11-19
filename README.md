# LLM-as-a-Judge for Conversational Music Recommendation
- **arXiv:** http://arxiv.org/abs/2607.25640
- **Conference:** [20th ACM Conference on Recommender Systems (RecSys '26)](https://recsys.acm.org/recsys26/)


**LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation**

> Seungheon Doh, Bruno Sguerra, Sergio Oramas, Elena V. Epure, and Juhan Nam.

> Accepted as a short paper at the **20th ACM Conference on Recommender Systems (RecSys '26)**, held September 28–October 2, 2026, in Minneapolis, Minnesota, USA.

## Overview

Conversational recommender systems must not only retrieve relevant items but also generate responses that explain and personalize their recommendations. Traditional reference-based text metrics are poorly suited to this setting because many valid responses may differ substantially from a single reference and because response quality depends on the full interaction context.

This work studies whether an LLM can serve as a reliable evaluator of conversational music recommendation responses. Our framework evaluates two independent dimensions:

- **Personalization Quality:** how well a response reflects the user's request, conversation history, and profile.
- **Explanation Quality:** how clearly the response connects the recommended item and its musical attributes to the user's needs.

The study covers 20 multi-turn recommendation sessions and responses from four open-source instruction-tuned models. We collect 400 ratings from 20 music-domain experts and compare human judgments with reference-based metrics and contextual LLM judges. The results show that LLM judges align more closely with human assessments than traditional reference-based metrics, while the remaining moderate correlation indicates that human oversight is still important.

The released pipeline has two stages:

1. Generate conversational recommendation responses with open-source models.
2. Evaluate each response using an LLM judge conditioned on the user profile, full dialogue context, recommended-track metadata, and the original domain-specific rubric.

## Setup

Python 3.10 or newer and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Google Colab Notebook
A ready-to-run Google Colab notebook is provided at [`llm_as_a_judge_for_crs.ipynb`](file:///Users/ibanana/Documents/coding/coding_kuliah/kapita/llm-as-a-judge-for-crs/llm_as_a_judge_for_crs.ipynb) with:
- Automated GPU environment setup & model inference.
- Option to restore already-generated responses from Google Drive.
- Multi-API-key rotation with automatic rate pacing.
- Automated backup of all experimental artifacts to Google Drive.

The scripts load these TalkPlay datasets from Hugging Face:

- `talkpl-ai/TalkPlayData-Challenge-Blind-A`
- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`
- `talkpl-ai/TalkPlayData-Challenge-User-Metadata`

Make sure they are accessible before running the pipeline.

## 1. Multi-turn Filtering & Response Generation

The script automatically filters out non-multi-turn sessions (sessions with fewer than 2 user turns) and saves the selected session IDs to `exp/filtered_session_ids.json` (and `exp/filtered_session_ids.txt`).

Run all four response models:

```bash
bash scripts/infer.sh
```
*(Automatically applies `--load_in_4bit` for larger models like `google/gemma-4-E4B-it` to fit in consumer/Colab T4 memory, and skips already-completed sessions).*

Or run an individual model (e.g. 20 multi-turn sessions):

```bash
python generate_response.py \
  --llm_model meta-llama/Llama-3.2-1B-Instruct \
  --num_sessions 20 \
  --device cuda
```

Use `--load_in_4bit` for large models to reduce VRAM from 16 GB to ~4.5 GB:

```bash
python generate_response.py \
  --llm_model google/gemma-4-E4B-it \
  --device cuda \
  --load_in_4bit
```

Responses are saved to:

```text
exp/response/{session_id}/{model_name}.txt
```

## 2. Run the conversational LLM-as-a-Judge (Multi-Key Rotation)

Set single or multiple Gemini API keys for automatic round-robin rotation, 4s rate pacing, and 60s cooldown on 429/503:

```bash
export GEMINI_API_KEYS="your_key_1,your_key_2,your_key_3"
```

Run the judge evaluation:

```bash
python eval_llm_as_a_judge.py \
  --judge_model gemini-3.8-flash
```

Or evaluate a specific subset of target models:

```bash
python eval_llm_as_a_judge.py \
  --judge_model gemini-3.8-flash \
  --target_models meta-llama-Llama-3.2-1B-Instruct meta-llama-Llama-3.2-3B-Instruct google-gemma-4-E2B-it
```

Outputs are saved without score post-processing:

```text
exp/judge_score/{judge_model}/{session_id}/{response_model}.json
```

## 3. Summarize Metrics & Export

Aggregate evaluation scores across all models and export consolidated data:

```bash
python summarize_results.py --judge_model gemini-3.8-flash
```

This generates:
- `exp/evaluation_summary.csv` — Average Personalization & Explanation scores per model.
- `exp/consolidated_evaluation_results.json` — Consolidated records containing session ID, model, prompt context, generated response, ratings, and judge rationales.

## Output schema

```json
{
  "personalization_score": int,
  "personalization_reason": str,
  "explanation_score": int,
  "explanation_reason": str
}
```

API keys, datasets, generated responses, and judge outputs are not included in the repository.


## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{doh2026llmjudge,
  author    = {Doh, Seungheon and Sguerra, Bruno and Oramas, Sergio and Epure, Elena V. and Nam, Juhan},
  title     = {{LLM-as-a-Judge} for Evaluating System Responses in Conversational Music Recommendation},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  series    = {RecSys '26},
  year      = {2026},
  location  = {Minneapolis, MN, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  note      = {Accepted short paper. DOI and page numbers forthcoming}
}
```
