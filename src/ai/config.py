import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    llm_model: str = "deepseek/deepseek-chat"
    llm_api_key: str = ""
    llm_fallback_models: list[str] = None

    tavily_api_key: str = ""

    def __post_init__(self):
        if self.llm_fallback_models is None:
            self.llm_fallback_models = []

    @classmethod
    def from_env(cls, env_path: str = None) -> "Config":
        load_dotenv(dotenv_path=env_path)

        api_key = ""
        model = os.getenv("LLM_MODEL", "")

        keys = {
            "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "gemini": os.getenv("GEMINI_API_KEY", ""),
        }

        if not model:
            for provider, key in keys.items():
                if key:
                    api_key = key
                    model = f"{provider}/deepseek-chat" if provider == "deepseek" else f"{provider}/gpt-4o" if provider == "openai" else f"{provider}/gemini-2.0-flash"
                    break

        if model and not api_key:
            for prefix in ("deepseek/", "openai/", "gemini/"):
                if model.startswith(prefix):
                    provider = prefix.rstrip("/")
                    api_key = keys.get(provider, "")
                    break

        if not api_key:
            for key in keys.values():
                if key:
                    api_key = key
                    break

        return cls(
            llm_model=model or "deepseek/deepseek-chat",
            llm_api_key=api_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )
