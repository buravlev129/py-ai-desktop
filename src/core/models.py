"""
Рекомендуемая архитектура проекта
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from pydantic import BaseModel

# ─── Общие модели данных ───

class Message(BaseModel):
    role: str          # "system" | "user" | "assistant"
    content: str

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict | None = None
    raw: dict = {}     # оригинальный ответ для отладки


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    extra_params: dict = field(default_factory=dict)


class BaseLLMConnector(ABC):
    """Базовый класс для всех LLM"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=self._build_headers()
        )

    @abstractmethod
    def _build_headers(self) -> dict:
        ...

    @abstractmethod
    def _build_payload(self, messages: list[Message]) -> dict:
        ...

    @abstractmethod
    def _parse_response(self, raw: dict) -> LLMResponse:
        ...

    async def chat(self, messages: list[Message]) -> LLMResponse:
        payload = self._build_payload(messages)
        response = await self.client.post(
            self._endpoint(),
            json=payload
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def _endpoint(self) -> str:
        return "/chat/completions"

    async def close(self):
        await self.client.aclose()


