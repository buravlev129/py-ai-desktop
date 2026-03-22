"""
Function Calling / Tools система для LLM.

Позволяет:
- Регистрировать Python-функции как инструменты
- Автоматически генерировать JSON Schema
- Вызывать функции по запросу LLM
"""
from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Optional, Union, 
    get_type_hints, get_origin, get_args
)
from pydantic import BaseModel
from docstring_parser import parse as parse_docstring


# ─── Модели данных ───

class ToolParameter(BaseModel):
    """Параметр инструмента"""
    type: str
    description: str = ""
    enum: Optional[list[str]] = None
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Схема инструмента для отправки в LLM"""
    name: str
    description: str
    parameters: dict[str, dict]


class ToolCall(BaseModel):
    """Вызов инструмента от LLM"""
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Результат выполнения инструмента"""
    call_id: str
    name: str
    result: Any
    error: Optional[str] = None
    success: bool = True


# ─── Базовый класс для инструментов ───

class BaseTool(ABC):
    """Абстрактный базовый класс для инструментов"""
    
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Выполнить инструмент"""
        ...
    
    def to_openai_schema(self) -> dict:
        """Преобразовать в схему OpenAI"""
        properties = {}
        required = []
        
        for param_name, param in self.parameters.items():
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum  # type: ignore
            if param.default is None:
                required.append(param_name)
            properties[param_name] = prop
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


# ─── Python-функция как инструмент ───

class FunctionTool(BaseTool):
    """
    Обёртка над Python-функцией для использования как инструмент LLM.
    
    Автоматически извлекает:
    - Имя функции
    - Описание из docstring
    - Параметры из type hints и docstring
    
    Usage:
        @tool
        async def get_weather(city: str) -> str:
            '''Get current weather for a city.
            
            Args:
                city: Name of the city
            '''
            return f"Weather in {city}: sunny"
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or self._extract_description()
        self.parameters = self._extract_parameters()
        self._is_async = asyncio.iscoroutinefunction(func)
    
    def _extract_description(self) -> str:
        """Извлечь описание из docstring"""
        docstring = self.func.__doc__
        if not docstring:
            return f"Execute {self.name}"
        
        parsed = parse_docstring(docstring)
        return parsed.short_description or parsed.description or f"Execute {self.name}"
    
    def _extract_parameters(self) -> dict[str, ToolParameter]:
        """Извлечь параметры из type hints и docstring"""
        params = {}
        
        # Получаем type hints
        hints = get_type_hints(self.func)
        
        # Парсим docstring для описаний параметров
        docstring_params = {}
        if self.func.__doc__:
            parsed = parse_docstring(self.func.__doc__)
            for p in parsed.params:
                docstring_params[p.arg_name] = p.description or ""
        
        # Получаем сигнатуру функции
        sig = inspect.signature(self.func)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Тип
            type_hint = hints.get(param_name, str)
            param_type = self._python_type_to_json(type_hint)
            
            # Описание
            description = docstring_params.get(param_name, "")
            
            # Default
            default = None if param.default is inspect.Parameter.empty else param.default
            
            params[param_name] = ToolParameter(
                type=param_type,
                description=description,
                default=default,
            )
        
        return params
    
    def _python_type_to_json(self, python_type: type) -> str:
        """Преобразовать Python тип в JSON Schema тип"""
        origin = get_origin(python_type)
        
        if origin is list or python_type is list:
            return "array"
        if origin is dict or python_type is dict:
            return "object"
        if python_type is bool:
            return "boolean"
        if python_type is int:
            return "integer"
        if python_type is float:
            return "number"
        
        return "string"
    
    async def execute(self, **kwargs) -> Any:
        """Выполнить функцию"""
        if self._is_async:
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)


# ─── Декоратор для создания инструментов ───

def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[FunctionTool, Callable]:
    """
    Декоратор для создания инструмента из функции.
    
    Usage:
        @tool
        def my_function(arg: str) -> str:
            '''Description'''
            return arg
        
        @tool(name="custom_name")
        async def another_function(x: int) -> int:
            return x * 2
    """
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description)
    
    if func is not None:
        return decorator(func)
    return decorator


# ─── Реестр инструментов ───

class ToolRegistry:
    """
    Реестр для управления инструментами.
    
    Features:
    - Регистрация инструментов
    - Генерация схем для API
    - Выполнение инструментов по имени
    - Поддержка синхронных и асинхронных функций
    
    Usage:
        registry = ToolRegistry()
        
        @registry.register
        def get_weather(city: str) -> str:
            '''Get weather for a city'''
            return "sunny"
        
        # Или явно
        registry.register_tool(my_tool)
        
        # Получить схемы для API
        schemas = registry.get_schemas()
        
        # Выполнить инструмент
        result = await registry.execute("get_weather", {"city": "Moscow"})
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, func: Callable) -> Callable:
        """Декоратор для регистрации функции как инструмента"""
        tool_obj = FunctionTool(func)
        self._tools[tool_obj.name] = tool_obj
        return func
    
    def register_tool(self, tool: BaseTool) -> None:
        """Зарегистрировать инструмент явно"""
        self._tools[tool.name] = tool
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Зарегистрировать функцию как инструмент"""
        tool_obj = FunctionTool(func, name=name, description=description)
        self._tools[tool_obj.name] = tool_obj
    
    def unregister(self, name: str) -> bool:
        """Удалить инструмент из реестра"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Получить инструмент по имени"""
        return self._tools.get(name)
    
    def get_schemas(self) -> list[dict]:
        """Получить схемы всех инструментов для API (OpenAI format)"""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def get_tool_names(self) -> list[str]:
        """Получить имена всех инструментов"""
        return list(self._tools.keys())
    
    async def execute(self, name: str, arguments: dict) -> Any:
        """Выполнить инструмент по имени"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return await tool.execute(**arguments)
    
    async def execute_call(self, call: ToolCall) -> ToolResult:
        """Выполнить вызов инструмента от LLM"""
        try:
            result = await self.execute(call.name, call.arguments)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=result,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                result=None,
                error=str(e),
                success=False,
            )
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)


# ─── Встроенные инструменты ───

class CalculatorTool(BaseTool):
    """Калькулятор для математических выражений"""
    
    name = "calculator"
    description = "Evaluate a mathematical expression. Use for calculations."
    parameters = {
        "expression": ToolParameter(
            type="string",
            description="Mathematical expression to evaluate, e.g. '2 + 2 * 3'",
        )
    }
    
    async def execute(self, expression: str) -> float:        # type: ignore
        # Безопасное вычисление (только базовые операции)
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid characters in expression")
        return eval(expression)


class CurrentTimeTool(BaseTool):
    """Получение текущего времени"""
    
    name = "get_current_time"
    description = "Get the current date and time"
    parameters = {}
    
    async def execute(self) -> str:           # type: ignore
        from datetime import datetime
        return datetime.now().isoformat()

