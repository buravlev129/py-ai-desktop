"""
Базовые модели и абстрактные классы для LLM коннекторов.
"""
from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any, Union, List

import httpx
from pydantic import BaseModel, Field

# ─── Общие модели данных ───

class Message(BaseModel):
    """Сообщение в диалоге"""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[List["ToolCallData"]] = None
    tool_call_id: Optional[str] = None  # для role="tool"
    name: Optional[str] = None  # имя инструмента для role="tool"
    
    class Config:
        # Разрешаем дополнительные поля
        extra = "allow"


class ToolCallData(BaseModel):
    """Данные о вызове инструмента в ответе LLM"""
    id: str
    type: str = "function"
    function: "FunctionCallData"


class FunctionCallData(BaseModel):
    """Данные о вызове функции"""
    name: str
    arguments: str  # JSON строка с аргументами


class LLMResponse(BaseModel):
    """Ответ от LLM"""
    content: Optional[str] = None
    model: str
    usage: dict | None = None
    raw: dict = {}     # оригинальный ответ для отладки
    tool_calls: Optional[List[ToolCallData]] = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Проверяет, есть ли вызовы инструментов"""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def get_tool_calls_as_dicts(self) -> List[dict]:
        """Получить tool_calls в виде словарей"""
        if not self.tool_calls:
            return []
        return [tc.model_dump() for tc in self.tool_calls]


@dataclass
class RetrySettings:
    """Настройки retry для запросов"""
    enabled: bool = True
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    extra_params: dict = field(default_factory=dict)
    retry: Optional[RetrySettings] = field(default_factory=RetrySettings)


class StreamChunk(BaseModel):
    """Часть потокового ответа"""
    content: str
    model: str | None = None
    finish_reason: str | None = None


class BaseLLMConnector(ABC):
    """Базовый класс для всех LLM"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=self._build_headers()
        )
        # Инициализируем retry handler если включен
        self._retry_handler = None
        if config.retry and config.retry.enabled:
            from .retry import RetryHandler, RetryConfig
            retry_config = RetryConfig(
                max_retries=config.retry.max_retries,
                base_delay=config.retry.base_delay,
                max_delay=config.retry.max_delay,
                exponential_base=config.retry.exponential_base,
                jitter=config.retry.jitter,
            )
            self._retry_handler = RetryHandler(retry_config)

    @abstractmethod
    def _build_headers(self) -> dict:
        ...

    @abstractmethod
    def _build_payload(self, messages: list[Message]) -> dict:
        ...

    @abstractmethod
    def _parse_response(self, raw: dict) -> LLMResponse:
        ...

    @abstractmethod
    async def chat_stream(self, messages: list[Message]) -> AsyncIterator[StreamChunk]:
        """
        Потоковая генерация ответа.
        Возвращает асинхронный итератор с чанками ответа.
        """
        ...

    async def chat(
        self, 
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> LLMResponse:
        """
        Обычный (не потоковый) запрос с поддержкой retry и tools.
        
        Args:
            messages: Список сообщений
            tools: Список схем инструментов (OpenAI format)
            tool_choice: "auto", "none", "required", или {"type": "function", "function": {"name": "..."}}
        """
        if self._retry_handler:
            return await self._retry_handler.execute(
                self._chat_impl, messages, tools, tool_choice
            )
        return await self._chat_impl(messages, tools, tool_choice)
    
    async def _chat_impl(
        self, 
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> LLMResponse:
        """Внутренняя реализация chat без retry"""
        from .retry import parse_http_error
        
        payload = self._build_payload(messages)
        
        # Добавляем tools если переданы
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        
        try:
            response = await self.client.post(
                self._endpoint(),
                json=payload
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except httpx.HTTPStatusError as e:
            raise parse_http_error(e)

    async def chat_collect(self, messages: list[Message]) -> LLMResponse:
        """
        Потоковый запрос, но собирает весь ответ в один LLMResponse.
        Удобно, когда нужен полный ответ, но с прогрессом.
        """
        full_content = ""
        model = None
        usage = None

        async for chunk in self.chat_stream(messages): # type: ignore
            full_content += chunk.content
            if chunk.model:
                model = chunk.model

        return LLMResponse(
            content=full_content,
            model=model or self.config.model,
            usage=usage,
            raw={"streaming": True}
        )

    def _endpoint(self) -> str:
        return "/chat/completions"

    async def close(self):
        await self.client.aclose()

