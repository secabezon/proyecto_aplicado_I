from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from project_config import cfg

logger = logging.getLogger(__name__)


def _load_env_from_root() -> None:
    """
    Load environment variables from a .env file located at project root.

    Streamlit may change the working directory; use an absolute path to avoid missing vars.
    """
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


_load_env_from_root()


def _resolve_hf_token() -> str:
    """
    Resolve the Hugging Face access token.

    Priority:
        1) cfg.HF_TOKEN
        2) environment variable HF_TOKEN (including .env)

    Returns:
        Hugging Face token.

    Raises:
        ValueError: If the token is missing.
    """
    token_from_cfg = (getattr(cfg, "HF_TOKEN", "") or "").strip()
    if token_from_cfg:
        return token_from_cfg

    token_from_env = (os.getenv("HF_TOKEN") or "").strip()
    if token_from_env:
        return token_from_env

    raise ValueError(
        "Missing HF token. Provide HF_TOKEN in config/config.py or set HF_TOKEN in .env."
    )


def llm_prompt(
    messages: List[Dict[str, str]],
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call a chat-capable model using Hugging Face InferenceClient.

    Args:
        messages: Chat messages in OpenAI-compatible format.
        temperature: Sampling temperature.
        response_format: Optional structured output request (provider/model dependent).

    Returns:
        The assistant text output.

    Raises:
        ValueError: If the provider/model combination is not available.
    """
    from huggingface_hub import InferenceClient  # type: ignore
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore

    token = _resolve_hf_token()

    model = (getattr(cfg, "HF_CHAT_MODEL", "") or "").strip()
    if not model:
        raise ValueError("Missing HF_CHAT_MODEL in config/config.py")

    preferred_provider = (getattr(cfg, "HF_PROVIDER", "auto") or "auto").strip()
    max_tokens = int(getattr(cfg, "LLM_MAX_TOKENS", 900))
    timeout_s = float(getattr(cfg, "HF_TIMEOUT_SECONDS", 120.0))

    providers_to_try = [preferred_provider]
    if preferred_provider.lower() != "auto":
        providers_to_try.append("auto")

    last_error: Optional[Exception] = None

    for provider in providers_to_try:
        t0 = time.perf_counter()
        try:
            client = InferenceClient(provider=provider, token=token, timeout=timeout_s)

            kwargs: Dict[str, Any] = {}
            if response_format is not None:
                kwargs["response_format"] = response_format

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = completion.choices[0].message.content
            out = (content or "").strip()
            dt = time.perf_counter() - t0
            logger.info(
                "LLM call completed | provider=%s | model=%s | seconds=%.2f | chars=%d",
                provider,
                model,
                dt,
                len(out),
            )
            return out

        except HfHubHTTPError as exc:
            last_error = exc
            dt = time.perf_counter() - t0
            logger.warning(
                "LLM call failed | provider=%s | model=%s | seconds=%.2f | error=%s",
                provider,
                model,
                dt,
                str(exc),
            )
            message = str(exc)
            if "404" in message or "Not Found" in message:
                continue
            raise

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            dt = time.perf_counter() - t0
            logger.warning(
                "LLM call failed | provider=%s | model=%s | seconds=%.2f | error=%s",
                provider,
                model,
                dt,
                str(exc),
            )
            continue

    raise ValueError(
        f"HF chat call failed (model='{model}', providers tried={providers_to_try})."
    ) from last_error


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse a JSON string defensively.

    Args:
        text: Raw JSON text.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If parsing fails.
    """
    cleaned = (text or "").strip()
    return json.loads(cleaned)
