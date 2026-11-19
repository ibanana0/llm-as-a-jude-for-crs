"""
Script to aggregate LLM-as-a-Judge evaluation results, compute summary metrics,
and export consolidated experiment data (conversations, responses, scores).
"""

import argparse
import json
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SESSION_FILE = BASE_DIR / "exp" / "filtered_session_ids.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize and export evaluation results.")
    parser.add_argument(
        "--judge_model",
        default="gemini-2.5-flash",
        help="Gemini judge model identifier used during evaluation.",
    )
    parser.add_argument(
        "--session_ids_path",
        default=str(DEFAULT_SESSION_FILE),
        help="Path to filtered session IDs file.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(BASE_DIR / "exp"),
        help="Directory to save summary and consolidated export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    judge_tag = args.judge_model.replace("/", "-")
    judge_dir = BASE_DIR / "exp" / "judge_score" / judge_tag
    response_dir = BASE_DIR / "exp" / "response"

    session_file = Path(args.session_ids_path)
    if session_file.is_file():
        with open(session_file, "r", encoding="utf-8") as f:
            session_ids = json.load(f)
    else:
        # Fallback to directories found in response_dir
        session_ids = [d.name for d in response_dir.iterdir() if d.is_dir()]

    print(f"Aggregating results for {len(session_ids)} sessions...")

    all_records = []
    model_scores = {}

    for sid in sorted(session_ids):
        sid_judge_dir = judge_dir / sid
        sid_resp_dir = response_dir / sid

        if not sid_judge_dir.is_dir():
            continue

        for score_file in sid_judge_dir.glob("*.json"):
            model_name = score_file.stem
            try:
                with open(score_file, "r", encoding="utf-8") as f:
                    score_data = json.load(f)
            except Exception as e:
                print(f"Error reading {score_file}: {e}")
                continue

            resp_file = sid_resp_dir / f"{model_name}.txt"
            resp_text = resp_file.read_text(encoding="utf-8") if resp_file.is_file() else ""

            pers_score = score_data.get("personalization_score")
            expl_score = score_data.get("explanation_score")

            record = {
                "session_id": sid,
                "model_name": model_name,
                "personalization_score": pers_score,
                "personalization_reason": score_data.get("personalization_reason", ""),
                "explanation_score": expl_score,
                "explanation_reason": score_data.get("explanation_reason", ""),
                "assistant_response": resp_text,
            }
            all_records.append(record)

            if model_name not in model_scores:
                model_scores[model_name] = {
                    "personalization": [],
                    "explanation": [],
                }
            if pers_score is not None:
                model_scores[model_name]["personalization"].append(pers_score)
            if expl_score is not None:
                model_scores[model_name]["explanation"].append(expl_score)

    if not all_records:
        print("No evaluation records found. Make sure eval_llm_as_a_judge.py has completed.")
        return

    # 1. Save detailed records to consolidated JSON
    consolidated_json = output_dir / "consolidated_evaluation_results.json"
    with open(consolidated_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"Saved full consolidated records to: {consolidated_json}")

    # 2. Compute averages per model
    summary_data = []
    print("\n" + "=" * 70)
    print(f"{'Model':<35} | {'Personalization (Avg)':<22} | {'Explanation (Avg)':<18}")
    print("-" * 70)

    for model, scores in sorted(model_scores.items()):
        p_list = scores["personalization"]
        e_list = scores["explanation"]
        p_avg = sum(p_list) / len(p_list) if p_list else 0.0
        e_avg = sum(e_list) / len(e_list) if e_list else 0.0

        summary_data.append({
            "model_name": model,
            "session_count": len(p_list),
            "avg_personalization_score": round(p_avg, 2),
            "avg_explanation_score": round(e_avg, 2),
        })
        print(f"{model:<35} | {p_avg:<22.2f} | {e_avg:<18.2f}")

    print("=" * 70 + "\n")

    # 3. Save summary CSV
    summary_csv = output_dir / "evaluation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "session_count", "avg_personalization_score", "avg_explanation_score"],
        )
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"Saved summary metrics CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
