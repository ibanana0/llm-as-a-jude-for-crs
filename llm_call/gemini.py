"""Google Gemini client with multi-key rotation and rate-limit pacing."""

from __future__ import annotations

import os
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
    """Manages a pool of Gemini API keys with round-robin rotation, cooldowns, and pacing."""

    def __init__(self, api_keys: List[str], min_interval_per_key: float = 4.0):
        if not api_keys:
            raise ValueError("No Gemini API keys provided.")
        self.keys = api_keys
        self.min_interval = min_interval_per_key
        self.clients = [genai.Client(api_key=k) for k in self.keys]
        self.last_used = [0.0] * len(self.keys)
        self.cooldown_until = [0.0] * len(self.keys)
        self.current_idx = 0

    def get_client(self) -> tuple[genai.Client, int]:
        now = time.time()
        num_keys = len(self.keys)

        # Check for any active key not in cooldown
        for attempt in range(num_keys):
            idx = (self.current_idx + attempt) % num_keys
            if now >= self.cooldown_until[idx]:
                self.current_idx = (idx + 1) % num_keys
                # Enforce pacing for this key if needed
                elapsed = now - self.last_used[idx]
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_used[idx] = time.time()
                return self.clients[idx], idx

        # If all keys are in cooldown, sleep until the earliest one recovers
        min_cooldown = min(self.cooldown_until)
        wait_seconds = max(1.0, min_cooldown - now + 1.0)
        print(f"\n  [All {num_keys} API keys in cooldown] Waiting {wait_seconds:.1f}s for key recovery...")
        time.sleep(wait_seconds)
        return self.get_client()

    def mark_throttled(self, key_idx: int, cooldown_seconds: float = 60.0):
        self.cooldown_until[key_idx] = time.time() + cooldown_seconds
        key_str = self.keys[key_idx]
        masked_key = (key_str[:6] + "..." + key_str[-4:]) if len(key_str) > 10 else "***"
        print(f"\n  ⚠️ Key #{key_idx + 1} ({masked_key}) throttled. Cooldown set to {cooldown_seconds}s. Rotating to next key...")


class Gemini_LLM_Client:
    """Gemini text generation client with schema-constrained output and multi-key rotation."""

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        api_keys: Optional[Union[str, List[str]]] = None,
        api_key: Optional[str] = None,
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

        self.pool = GeminiKeyPool(key_list)
        print(f"Initialized Gemini Client with {len(key_list)} API key(s) for model: {self.model_name}")

    def chat_completion(self, query: str, schema_type: str = "judge", max_key_retries: int = 5) -> str:
        if schema_type == "judge":
            response_schema = JudgeScore
        else:
            response_schema = None

        last_exc = None
        attempts = max(max_key_retries, len(self.pool.keys) * 2)
        for _ in range(attempts):
            client, key_idx = self.pool.get_client()
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
                return model_output_text.text

            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                # Check for rate limit or server capacity issues (429, 503, quota, resource_exhausted, high demand)
                if any(term in err_str for term in ["429", "503", "resource_exhausted", "high demand", "unavailable", "quota", "rate"]):
                    self.pool.mark_throttled(key_idx)
                    continue
                else:
                    raise exc

        raise RuntimeError(f"All API keys failed or exhausted: {last_exc}") from last_exc
