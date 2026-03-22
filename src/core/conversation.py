"""
Менеджер диалога (Conversation Manager).
Управление историей сообщений и контекстом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Callable
import json

from .models import Message, LLMResponse

if TYPE_CHECKING:
    from .models import BaseLLMConnector


@dataclass
class ConversationConfig:
    """Конфигурация диалога"""
    max_messages: int = 50  # максимум сообщений в истории
    max_tokens: int = 4000  # лимит токенов (для truncate)
    system_prompt: str = "Ты полезный помощник."
    auto_save: bool = False
    save_path: Optional[str] = None


@dataclass
class Conversation:
    """
    Управление историей диалога.
    
    Features:
    - Добавление сообщений пользователя и ассистента
    - Автоматическое управление system prompt
    - Обрезка истории при превышении лимита
    - Сохранение/загрузка в JSON
    - Подсчёт сообщений
    
    Usage:
        conv = Conversation(system_prompt="Ты эксперт по Python")
        conv.add_user_message("Как создать список?")
        response = await llm.chat(conv.get_messages())
        conv.add_assistant_message(response.content)
    """
    
    messages: List[Message] = field(default_factory=list)
    config: ConversationConfig = field(default_factory=ConversationConfig)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Инициализация после создания"""
        if isinstance(self.config, dict):
            self.config = ConversationConfig(**self.config)
    
    # ─── Основные методы ───
    
    def add_user_message(self, content: str) -> Message:
        """Добавить сообщение пользователя"""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        self._update()
        return msg
    
    def add_assistant_message(self, content: str) -> Message:
        """Добавить сообщение ассистента"""
        msg = Message(role="assistant", content=content)
        self.messages.append(msg)
        self._update()
        return msg
    
    def add_message(self, role: str, content: str) -> Message:
        """Добавить сообщение с произвольной ролью"""
        if role == "system":
            raise ValueError("Use set_system_prompt() for system messages")
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self._update()
        return msg
    
    def set_system_prompt(self, prompt: str):
        """Установить system prompt"""
        self.config.system_prompt = prompt
        self._update()
    
    def get_messages(self) -> List[Message]:
        """
        Получить все сообщения для отправки в LLM.
        Включает system prompt в начале.
        """
        return [
            Message(role="system", content=self.config.system_prompt),
            *self.messages
        ]
    
    def get_last_message(self) -> Optional[Message]:
        """Получить последнее сообщение"""
        return self.messages[-1] if self.messages else None
    
    def get_last_user_message(self) -> Optional[Message]:
        """Получить последнее сообщение пользователя"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None
    
    # ─── Управление историей ───
    
    def truncate(self, keep_last: int = 2) -> int:
        """
        Обрезка истории, сохраняя последние N обменов.
        Один обмен = user + assistant сообщение.
        
        Returns: количество удалённых сообщений
        """
        if len(self.messages) <= keep_last * 2:
            return 0
        
        # Сохраняем последние N обменов (2*N сообщений)
        messages_to_keep = keep_last * 2
        removed = len(self.messages) - messages_to_keep
        self.messages = self.messages[-messages_to_keep:]
        self._update()
        
        return removed
    
    def truncate_to_max_messages(self) -> int:
        """
        Обрезка истории до max_messages.
        
        Returns: количество удалённых сообщений
        """
        if len(self.messages) <= self.config.max_messages:
            return 0
        
        removed = len(self.messages) - self.config.max_messages
        self.messages = self.messages[-self.config.max_messages:]
        self._update()
        
        return removed
    
    def clear(self):
        """Очистить историю (сохраняя system prompt)"""
        self.messages.clear()
        self._update()
    
    # ─── Статистика ───
    
    @property
    def message_count(self) -> int:
        """Количество сообщений в истории (без system)"""
        return len(self.messages)
    
    @property
    def turn_count(self) -> int:
        """Количество полных обменов (вопрос-ответ)"""
        return sum(1 for m in self.messages if m.role == "user")
    
    def get_stats(self) -> dict:
        """Получить статистику диалога"""
        user_msgs = sum(1 for m in self.messages if m.role == "user")
        assistant_msgs = sum(1 for m in self.messages if m.role == "assistant")
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "turns": user_msgs,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    # ─── Сохранение/Загрузка ───
    
    def to_dict(self) -> dict:
        """Сериализация в словарь"""
        return {
            "messages": [m.model_dump() for m in self.messages],
            "config": {
                "max_messages": self.config.max_messages,
                "max_tokens": self.config.max_tokens,
                "system_prompt": self.config.system_prompt,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def to_json(self) -> str:
        """Сериализация в JSON строку"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Десериализация из словаря"""
        config = ConversationConfig(
            max_messages=data.get("config", {}).get("max_messages", 50),
            max_tokens=data.get("config", {}).get("max_tokens", 4000),
            system_prompt=data.get("config", {}).get("system_prompt", "Ты полезный помощник."),
        )
        
        messages = [Message(**m) for m in data.get("messages", [])]
        
        conv = cls(messages=messages, config=config)
        
        if "created_at" in data:
            conv.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            conv.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return conv
    
    @classmethod
    def from_json(cls, json_str: str) -> "Conversation":
        """Десериализация из JSON строки"""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Optional[str] = None):
        """Сохранить диалог в файл"""
        path = path or self.config.save_path
        if not path:
            raise ValueError("No save path specified")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "Conversation":
        """Загрузить диалог из файла"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
    
    # ─── Вспомогательные методы ───
    
    def _update(self):
        """Обновить метку времени"""
        self.updated_at = datetime.now()
        
        # Автосохранение
        if self.config.auto_save and self.config.save_path:
            self.save()


class ConversationManager:
    """
    Менеджер для управления несколькими диалогами.
    Полезно для multi-user приложений.
    """
    
    def __init__(self):
        self._conversations: dict[str, Conversation] = {}
    
    def create(
        self,
        id: str,
        system_prompt: str = "Ты полезный помощник.",
        **config_kwargs
    ) -> Conversation:
        """Создать новый диалог"""
        config = ConversationConfig(system_prompt=system_prompt, **config_kwargs)
        conv = Conversation(config=config)
        self._conversations[id] = conv
        return conv
    
    def get(self, id: str) -> Optional[Conversation]:
        """Получить диалог по ID"""
        return self._conversations.get(id)
    
    def get_or_create(
        self,
        id: str,
        system_prompt: str = "Ты полезный помощник."
    ) -> Conversation:
        """Получить или создать диалог"""
        if id not in self._conversations:
            return self.create(id, system_prompt=system_prompt)
        return self._conversations[id]
    
    def delete(self, id: str) -> bool:
        """Удалить диалог"""
        if id in self._conversations:
            del self._conversations[id]
            return True
        return False
    
    def list_ids(self) -> List[str]:
        """Получить список ID всех диалогов"""
        return list(self._conversations.keys())
    
    def clear_all(self):
        """Удалить все диалоги"""
        self._conversations.clear()

