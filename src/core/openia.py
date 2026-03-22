"""
OpenAI-совместимый коннектор.
Работает для: OpenAI, DeepSeek, Qwen, Grok, Mistral и др.
Разница только в base_url и api_key.
"""
import json
from typing import AsyncIterator

from src.core.models import (
    BaseLLMConnector, Message, LLMResponse, StreamChunk,
    ToolCallData, FunctionCallData
)


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
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        # Специфичные параметры провайдера
        payload.update(self.config.extra_params)
        return payload

    def _parse_response(self, raw: dict) -> LLMResponse:
        choice = raw["choices"][0]
        message = choice["message"]
        
        # Парсим content
        content = message.get("content")
        
        # Парсим tool_calls если есть
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = []
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCallData(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=FunctionCallData(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    )
                ))
        
        return LLMResponse(
            content=content,
            model=raw.get("model", self.config.model),
            usage=raw.get("usage"),
            raw=raw,
            tool_calls=tool_calls,
        )

    async def chat_stream(self, messages: list[Message]) -> AsyncIterator[StreamChunk]:  # type: ignore
        """
        Потоковая генерация ответа для OpenAI-совместимых API.
        
        Формат SSE (Server-Sent Events):
        data: {"choices":[{"delta":{"content":"Hello"}}]}
        data: [DONE]
        """
        payload = self._build_payload(messages)
        payload["stream"] = True

        async with self.client.stream(
            "POST",
            self._endpoint(),
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                # Пропускаем пустые строки
                if not line or not line.strip():
                    continue
                
                # Пропускаем префикс "data: "
                if line.startswith("data: "):
                    data_str = line[6:]  # убираем "data: "
                    
                    # Конец потока
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason")
                            
                            if content or finish_reason:
                                yield StreamChunk(
                                    content=content,
                                    model=data.get("model"),
                                    finish_reason=finish_reason
                                )
                    except json.JSONDecodeError:
                        # Игнорируем некорректный JSON
                        continue
