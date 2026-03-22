"""
ToolExecutor - выполнение цикла вызова инструментов.

Управляет полным циклом:
1. Отправка запроса к LLM с tools
2. Выполнение tool_calls если есть
3. Отправка результатов обратно
4. Повторение до получения финального ответа
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, AsyncIterator

from .models import Message, LLMResponse, ToolCallData
from .tools import ToolRegistry, ToolCall, ToolResult

if TYPE_CHECKING:
    from .models import BaseLLMConnector


@dataclass
class ExecutorConfig:
    """Конфигурация executor"""
    max_iterations: int = 10  # максимум итераций tool calling
    verbose: bool = False  # выводить ли отладочную информацию


class ToolExecutor:
    """
    Управляет циклом вызова инструментов.
    
    Features:
    - Автоматическое выполнение tool calls
    - Ограничение на количество итераций
    - Поддержка streaming для финального ответа
    - Логирование
    
    Usage:
        executor = ToolExecutor(llm, tool_registry)
        
        # Простой вызов с автоматическим выполнением tools
        response = await executor.run(messages)
        
        # Streaming финального ответа
        async for chunk in executor.run_stream(messages):
            print(chunk.content, end="")
    """
    
    def __init__(
        self,
        llm: "BaseLLMConnector",
        tools: ToolRegistry,
        config: Optional[ExecutorConfig] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config or ExecutorConfig()
    
    async def run(
        self,
        messages: List[Message],
        tool_choice: Optional[str] = "auto",
    ) -> LLMResponse:
        """
        Выполнить запрос с автоматическим вызовом инструментов.
        
        Args:
            messages: История сообщений
            tool_choice: "auto", "none", "required"
        
        Returns:
            Финальный ответ от LLM (без tool calls)
        """
        messages = list(messages)  # копия
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            # Получаем схемы инструментов
            tool_schemas = self.tools.get_schemas()
            
            # Отправляем запрос
            response = await self.llm.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=tool_choice if tool_schemas else None,
            )
            
            # Если нет tool_calls — возвращаем ответ
            if not response.has_tool_calls:
                return response
            
            # Логируем
            if self.config.verbose:
                print(f"\n[Iteration {iteration}] LLM wants to call tools:")
                for tc in response.tool_calls:             # type: ignore
                    print(f"  - {tc.function.name}({tc.function.arguments})")
            
            # Добавляем сообщение ассистента с tool_calls
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))
            
            # Выполняем все tool calls
            for tool_call in response.tool_calls:           # type: ignore
                result = await self._execute_tool_call(tool_call)
                
                if self.config.verbose:
                    status = "✓" if result.success else "✗"
                    print(f"  {status} {tool_call.function.name}: {result.result or result.error}")
                
                # Добавляем результат tool call
                messages.append(Message(
                    role="tool",
                    content=json.dumps(result.result if result.success else {"error": result.error}, ensure_ascii=False),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                ))
        
        # Превысили лимит итераций
        raise RuntimeError(f"Exceeded max iterations ({self.config.max_iterations})")
    
    async def run_stream(
        self,
        messages: List[Message],
        tool_choice: Optional[str] = "auto",
    ) -> AsyncIterator[str]:
        """
        Выполнить запрос со streaming финального ответа.
        
        Tool calls выполняются без streaming,
        streaming работает только для финального текстового ответа.
        """
        messages = list(messages)
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            tool_schemas = self.tools.get_schemas()
            
            # Отправляем запрос (не streaming, т.к. нужно проверить tool_calls)
            response = await self.llm.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=tool_choice if tool_schemas else None,
            )
            
            if not response.has_tool_calls:
                # Финальный ответ — стримим
                async for chunk in self.llm.chat_stream(messages):       # type: ignore
                    if chunk.content:
                        yield chunk.content
                return
            
            # Выполняем tools
            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))
            
            for tool_call in response.tool_calls:       # type: ignore
                result = await self._execute_tool_call(tool_call)
                messages.append(Message(
                    role="tool",
                    content=json.dumps(result.result if result.success else {"error": result.error}, ensure_ascii=False),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                ))
        
        raise RuntimeError(f"Exceeded max iterations ({self.config.max_iterations})")
    
    async def _execute_tool_call(self, tool_call: ToolCallData) -> ToolResult:
        """Выполнить один tool call"""
        try:
            # Парсим аргументы
            arguments = json.loads(tool_call.function.arguments)
            
            # Создаём ToolCall для registry
            call = ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=arguments,
            )
            
            # Выполняем через registry
            return await self.tools.execute_call(call)
            
        except json.JSONDecodeError as e:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.function.name,
                result=None,
                error=f"Invalid JSON arguments: {e}",
                success=False,
            )
        except Exception as e:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.function.name,
                result=None,
                error=str(e),
                success=False,
            )
    
    async def close(self):
        """Закрыть соединение"""
        await self.llm.close()


# ─── Утилиты для создания сообщений ───

def create_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    result: Any,
    error: Optional[str] = None,
) -> Message:
    """
    Создать сообщение с результатом tool call.
    
    Args:
        tool_call_id: ID вызова инструмента
        tool_name: Имя инструмента
        result: Результат выполнения
        error: Ошибка если есть
    
    Returns:
        Message с role="tool"
    """
    content = json.dumps(result, ensure_ascii=False) if not error else json.dumps({"error": error})
    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
    )


def create_assistant_tool_call_message(
    tool_calls: List[ToolCallData],
    content: Optional[str] = None,
) -> Message:
    """
    Создать сообщение ассистента с tool calls.
    
    Args:
        tool_calls: Список вызовов инструментов
        content: Опциональный текст
    
    Returns:
        Message с role="assistant"
    """
    return Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )

