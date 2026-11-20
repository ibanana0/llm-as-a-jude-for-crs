"""Google Gemini client with multi-key rotation, round-robin turn-taking, and rate-limit pacing."""

from __future__ import annotations

import os
import re
import time
from typing import Optional, List, Union
from google import genai
import pydantic


class JudgeScore(pydantic.BaseModel):
    personalization_score: int = pydantic.Field(ge=1, le=5)
    personalization_reason: str
    explanation_score: int = pydantic.Field(ge=1, le=5)
    explanation_reason: str


class GeminiKeyPool:
    """Manages a pool of Gemini API keys with round-robin turn-taking, cooldowns, and pacing."""

    def __init__(self, api_keys: List[str], min_interval_per_key: float = 4.0):
        if not api_keys:
            raise ValueError("No Gemini API keys provided.")
        self.keys = api_keys
        self.min_interval = min_interval_per_key
        self.clients = [genai.Client(api_key=k) for k in self.keys]
        self.last_used = [0.0] * len(self.keys)
        self.cooldown_until = [0.0] * len(self.keys)
        self.current_idx = 0

    def get_masked_key(self, idx: int) -> str:
        key_str = self.keys[idx]
        if len(key_str) > 10:
            return f"{key_str[:6]}...{key_str[-4:]}"
        return "***"

    def get_client(self) -> tuple[genai.Client, int]:
        now = time.time()
        num_keys = len(self.keys)

        # Check for any active key not in cooldown, taking turns in round-robin order
        for attempt in range(num_keys):
            idx = (self.current_idx + attempt) % num_keys
            if now >= self.cooldown_until[idx]:
                # Advance pointer to take turn for next judge call
                self.current_idx = (idx + 1) % num_keys
                # Enforce pacing per key if needed (e.g. 15 RPM limit per individual key)
                elapsed = now - self.last_used[idx]
                if elapsed < self.min_interval:
                    wait_pacing = self.min_interval - elapsed
                    time.sleep(wait_pacing)
                self.last_used[idx] = time.time()
                return self.clients[idx], idx

        # If all keys are currently in cooldown, wait until the earliest one recovers
        min_cooldown = min(self.cooldown_until)
        earliest_idx = self.cooldown_until.index(min_cooldown)
        wait_seconds = max(1.0, min_cooldown - now + 1.0)
        print(
            f"\n  [All {num_keys} API keys in cooldown] Waiting {wait_seconds:.1f}s for Key #{earliest_idx + 1} "
            f"({self.get_masked_key(earliest_idx)}) to recover..."
        )
        time.sleep(wait_seconds)
        return self.get_client()

    def mark_throttled(self, key_idx: int, cooldown_seconds: float = 60.0):
        self.cooldown_until[key_idx] = time.time() + cooldown_seconds
        masked_key = self.get_masked_key(key_idx)
        print(
            f"\n  ⚠️ Key #{key_idx + 1} ({masked_key}) hit rate/quota limit. "
            f"Cooldown set to {cooldown_seconds:.0f}s. Rotating turn immediately to next key..."
        )


class Gemini_LLM_Client:
    """Gemini text generation client with schema-constrained output and multi-key rotation."""

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        api_keys: Optional[Union[str, List[str]]] = None,
        api_key: Optional[str] = None,
        min_interval_per_key: float = 4.0,
    ):
        self.model_name = model_name

        # Parse keys from arguments or environment
        raw_keys = api_keys or api_key or os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if isinstance(raw_keys, str):
            key_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
        elif isinstance(raw_keys, list):
            key_list = [k.strip() for k in raw_keys if isinstance(k, str) and k.strip()]
        else:
            key_list = []

        if not key_list:
            raise ValueError(
                "GEMINI_API_KEY or GEMINI_API_KEYS environment variable is not set."
            )

        self.pool = GeminiKeyPool(key_list, min_interval_per_key=min_interval_per_key)
        print(f"Initialized Gemini Client with {len(key_list)} API key(s) in rotation for model: {self.model_name}")

    def chat_completion(self, query: str, schema_type: str = "judge", max_key_retries: int = 5) -> str:
        if schema_type == "judge":
            response_schema = JudgeScore
        else:
            response_schema = None

        last_exc = None
        attempts = max(max_key_retries, len(self.pool.keys) * 3)
        for _ in range(attempts):
            client, key_idx = self.pool.get_client()
            masked_key = self.pool.get_masked_key(key_idx)
            print(f"  -> [Key Turn #{key_idx + 1}/{len(self.pool.keys)}: {masked_key}] Generating judge evaluation...")
            try:
                model_output_text = client.models.generate_content(
                    model=self.model_name,
                    contents=query,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                    }
                )
                if not model_output_text.text:
                    raise ValueError("Gemini returned an empty response.")
                self.pool.last_used[key_idx] = time.time()
                return model_output_text.text

            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

                # Check for rate limit or server capacity issues (429, 503, quota, resource_exhausted, high demand, TPM/RPM)
                is_rate_limit = (
                    status_code in (429, 503)
                    or any(
                        term in err_str
                        for term in [
                            "429",
                            "503",
                            "resource_exhausted",
                            "quota",
                            "rate",
                            "tpm",
                            "rpm",
                            "too many requests",
                            "high demand",
                            "unavailable",
                            "overloaded",
                            "capacity",
                            "exhausted",
                            "limit",
                        ]
                    )
                )

                if is_rate_limit:
                    cooldown = 60.0
                    match = re.search(r"retry (?:in|after) (\d+(?:\.\d+)?)s", err_str)
                    if match:
                        cooldown = float(match.group(1)) + 2.0
                    self.pool.mark_throttled(key_idx, cooldown_seconds=cooldown)
                    continue
                else:
                    raise exc

        raise RuntimeError(f"All API keys failed or exhausted: {last_exc}") from last_exc
