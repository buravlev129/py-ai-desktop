from src.core.models import BaseLLMConnector, Message, LLMResponse


# ───  ───
# Работает для: OpenAI, DeepSeek, Qwen, Grok, Mistral и др.

class OpenAICompatibleConnector(BaseLLMConnector):
    """
    OpenAI-совместимый коннектор.  
    Один класс для всех OpenAI-совместимых провайдеров.  
    OpenAI, DeepSeek, Qwen, Grok, Mistral и др.  
    Разница только в base_url и api_key.
    """

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message]) -> dict:
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        # Специфичные параметры провайдера
        payload.update(self.config.extra_params)
        return payload

    def _parse_response(self, raw: dict) -> LLMResponse:
        choice = raw["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=raw.get("model", self.config.model),
            usage=raw.get("usage"),
            raw=raw,
        )

