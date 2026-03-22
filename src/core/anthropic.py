from src.core.models import BaseLLMConnector, Message, LLMResponse


class AnthropicConnector(BaseLLMConnector):
    """
    Claude использует свой формат API.
    Нужен отдельный адаптер.
    """

    def _build_headers(self) -> dict:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _endpoint(self) -> str:
        return "/v1/messages"

    def _build_payload(self, messages: list[Message]) -> dict:
        # Claude: system — отдельное поле, НЕ в messages
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append(msg.model_dump())

        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if system_msg:
            payload["system"] = system_msg

        payload.update(self.config.extra_params)
        return payload

    def _parse_response(self, raw: dict) -> LLMResponse:
        return LLMResponse(
            content=raw["content"][0]["text"],
            model=raw.get("model", self.config.model),
            usage=raw.get("usage"),
            raw=raw,
        )


