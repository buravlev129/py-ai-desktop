"""
Anthropic (Claude) коннектор.
Claude использует свой формат API, отличный от OpenAI.
"""
import json
from typing import AsyncIterator

from src.core.models import (
    BaseLLMConnector, Message, LLMResponse, StreamChunk,
    ToolCallData, FunctionCallData
)


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
            elif msg.role == "tool":
                # Tool result для Claude
                chat_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }]
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant с tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    })
                chat_messages.append({"role": "assistant", "content": content})
            else:
                chat_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "max_tokens": self.config.max_tokens,
        }

        if system_msg:
            payload["system"] = system_msg

        # temperature для Claude опционален
        if self.config.temperature != 0.7:
            payload["temperature"] = self.config.temperature

        payload.update(self.config.extra_params)
        return payload

    def _parse_response(self, raw: dict) -> LLMResponse:
        """Парсинг ответа Anthropic"""
        content_blocks = raw.get("content", [])
        
        text_content = None
        tool_calls = []
        
        for block in content_blocks:
            if block["type"] == "text":
                text_content = block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCallData(
                    id=block["id"],
                    type="function",
                    function=FunctionCallData(
                        name=block["name"],
                        arguments=json.dumps(block["input"]),
                    )
                ))
        
        return LLMResponse(
            content=text_content,
            model=raw.get("model", self.config.model),
            usage=raw.get("usage"),
            raw=raw,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def chat_stream(self, messages: list[Message]) -> AsyncIterator[StreamChunk]:  # type: ignore
        """
        Потоковая генерация ответа для Anthropic API.
        
        Формат SSE Anthropic отличается от OpenAI:
        event: content_block_delta
        data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}
        
        event: message_stop
        data: {}
        """
        payload = self._build_payload(messages)
        payload["stream"] = True

        async with self.client.stream(
            "POST",
            self._endpoint(),
            json=payload
        ) as response:
            response.raise_for_status()
            
            current_event = None
            
            async for line in response.aiter_lines():
                line = line.strip()
                
                if not line:
                    continue
                
                # Парсим тип события
                if line.startswith("event: "):
                    current_event = line[7:]  # убираем "event: "
                    continue
                
                # Парсим данные события
                if line.startswith("data: "):
                    data_str = line[6:]
                    
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", current_event)
                        
                        # Основной тип для текста
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield StreamChunk(
                                        content=text,
                                        model=None
                                    )
                        
                        # Конец сообщения
                        elif event_type == "message_stop":
                            break
                        
                        # Ошибка
                        elif event_type == "error":
                            error = data.get("error", {})
                            raise Exception(f"Anthropic API error: {error.get('message', str(data))}")
                            
                    except json.JSONDecodeError:
                        continue

