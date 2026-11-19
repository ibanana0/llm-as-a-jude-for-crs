"""
Batch inference script for Music CRS.
"""

import argparse
import json
from pathlib import Path

from environments import MusicCatalog, UserProfileDB
from datasets import load_dataset
from tqdm import tqdm
from llm_call.hf import HF_LLM_Client


BASE_DIR = Path(__file__).resolve().parent

JUDGE_SESSION_IDS = [
  "525f9f69-31e5-4fee-9813-e903ad24e39e" # randomly selected session ID for evaluation
]
DEFAULT_SESSION_FILE = BASE_DIR / "exp" / "filtered_session_ids.json"


def is_multi_turn(session: dict, min_user_turns: int = 2) -> bool:
    """Filter out non-multi-turn sessions.
    
    A session is multi-turn if it has at least `min_user_turns` turns from the user.
    """
    conversations = session.get("conversations", [])
    user_turns = [turn for turn in conversations if turn.get("role") == "user"]
    return len(user_turns) >= min_user_turns


def filter_and_save_sessions(
    gt_dataset,
    min_user_turns: int = 2,
    num_sessions: int = 20,
    save_path: Path = DEFAULT_SESSION_FILE,
    force_filter: bool = False,
) -> list[str]:
    """Filter multi-turn sessions from dataset and save their session IDs to JSON and TXT files."""
    if save_path.is_file() and not force_filter:
        with save_path.open("r", encoding="utf-8") as f:
            session_ids = json.load(f)
        if session_ids:
            print(f"Loaded {len(session_ids)} existing multi-turn session IDs from {save_path}")
            return session_ids

    multi_turn_dataset = [item for item in gt_dataset if is_multi_turn(item, min_user_turns=min_user_turns)]
    print(f"Total sessions: {len(gt_dataset)} | Multi-turn sessions found: {len(multi_turn_dataset)}")

    if num_sessions > 0:
        selected = multi_turn_dataset[:num_sessions]
    else:
        selected = multi_turn_dataset

    session_ids = [item["session_id"] for item in selected]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(session_ids, f, indent=2)

    # Also save as plain text (one session_id per line)
    txt_path = save_path.with_suffix(".txt")
    txt_path.write_text("\n".join(session_ids) + "\n", encoding="utf-8")

    print(f"Saved {len(session_ids)} filtered session IDs to:\n  - {save_path}\n  - {txt_path}")
    return session_ids


def build_dialogue_context(conversations, music_db) -> str:
    """Build the model input from all turns preceding the target response."""
    dialogue_context = []
    for turn in conversations[:-1]:
        if turn["role"] == "music":
            role = "assistant"
            content = music_db.id_to_metadata(
                track_id=turn["content"],
            )
        else:
            role = turn["role"]
            content = turn["content"]
        dialogue_context.append({"role": role, "content": content})
    return dialogue_context


def main(args):
    prompt = """You are a music recommendation assistant. Generate an appropriate response about the track that has already been recommended, considering the recommended item, user demographics, and user query."""
    music_db = MusicCatalog(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=[
            "track_name",
            "artist_name",
            "album_name",
            "release_date",
            "tag_list",
        ],
    )
    user_db = UserProfileDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        split_types=["all_users"],
    )
    llm_agents = HF_LLM_Client(
        model_name=args.llm_model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    gt_dataset = load_dataset(
        args.dataset_name,
        split=args.split,
    )

    # Filter out non-multi-turn sessions and save session IDs
    save_path = Path(args.session_ids_path)
    session_ids = filter_and_save_sessions(
        gt_dataset=gt_dataset,
        min_user_turns=args.min_user_turns,
        num_sessions=args.num_sessions,
        save_path=save_path,
        force_filter=args.force_filter,
    )

    gt_dict = {item["session_id"]: item for item in gt_dataset}

    for sid in tqdm(sorted(session_ids), desc="Sessions"):
        model_name = args.llm_model.replace("/", "-")
        output_path = BASE_DIR / "exp" / "response" / sid / f"{model_name}.txt"
        if output_path.exists() and not args.overwrite:
            continue

        gt_instance = gt_dict[sid]
        conversations = gt_instance["conversations"]
        user_id = gt_instance["user_id"]
        user_profile_str = user_db.id_to_profile_str(user_id)
        dialogue_context = build_dialogue_context(conversations, music_db)
        messages = [{"role": "system", "content": prompt + "\n\n" + user_profile_str}]
        messages.extend(dialogue_context)
        _, model_output_text = llm_agents.chat_completion(messages, max_new_tokens=1024)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(model_output_text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Blind-A",
        help="Hugging Face dataset identifier (default: talkpl-ai/TalkPlayData-Challenge-Blind-A).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name (default: test).",
    )
    parser.add_argument(
        "--min_user_turns",
        type=int,
        default=2,
        help="Minimum number of user turns to classify as multi-turn (default: 2).",
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=20,
        help="Number of multi-turn sessions to evaluate (default: 20, paper setting). Use 0 for all.",
    )
    parser.add_argument(
        "--session_ids_path",
        type=str,
        default=str(DEFAULT_SESSION_FILE),
        help=f"Path to save/load filtered session IDs (default: {DEFAULT_SESSION_FILE}).",
    )
    parser.add_argument(
        "--force_filter",
        action="store_true",
        help="Force re-filtering even if saved session IDs file already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated response files.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization using bitsandbytes (saves RAM/VRAM for large models).",
    )
    args = parser.parse_args()
    main(args)
