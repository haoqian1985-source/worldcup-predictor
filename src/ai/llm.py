import logging
from typing import Optional

try:
    import litellm
    from litellm import completion as litellm_completion
except ImportError:
    litellm = None
    litellm_completion = None

from src.ai.config import Config

logger = logging.getLogger(__name__)


def call_llm(
    messages: list[dict],
    config: Config,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """Call LLM via LiteLLM with fallback support."""
    if litellm is None:
        raise ImportError("litellm is not installed. Run: pip install litellm")

    full_messages = list(messages)
    if system_prompt:
        full_messages.insert(0, {"role": "system", "content": system_prompt})

    models_to_try = [config.llm_model] + (config.llm_fallback_models or [])
    models_to_try = [m for m in models_to_try if m]

    last_error = None
    for model in models_to_try:
        try:
            kwargs = {
                "model": model,
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if config.llm_api_key:
                kwargs["api_key"] = config.llm_api_key

            response = litellm_completion(**kwargs)
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    "LLM %s | prompt=%d completion=%d total=%d",
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )
            return text

        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s", model, e)
            continue

    raise RuntimeError(f"All LLM models failed. Last error: {last_error}")
