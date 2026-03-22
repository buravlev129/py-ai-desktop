import os
from src.core.models import BaseLLMConnector, Message, LLMConfig, LLMResponse
from src.core.openia import OpenAICompatibleConnector
from src.core.anthropic import AnthropicConnector


class LLMFactory:
    """Создание коннектора по имени провайдера"""

    PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "connector": OpenAICompatibleConnector,
            "default_model": "",
        },

        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "connector": OpenAICompatibleConnector,
            "default_model": "deepseek-chat",
        },

        # https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
        "qwen": {
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "connector": OpenAICompatibleConnector,
            "default_model": "qwen-plus",
        }, 

        "grok": {
            "base_url": "https://api.x.ai/v1",
            "connector": OpenAICompatibleConnector,
            "default_model": "",
        },

        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "connector": AnthropicConnector,
            "default_model": "",
        },
    }

    @classmethod
    def create(
        cls,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs
    ) -> BaseLLMConnector:

        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        info = cls.PROVIDERS[provider]
        env_key = f"{provider.upper()}_API_KEY"

        config = LLMConfig(
            api_key=api_key or os.environ[env_key],
            model=model or info.get('default_model', ''),
            base_url=info["base_url"],
            **kwargs,
        )

        return info["connector"](config)

