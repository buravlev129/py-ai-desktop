"""
Примеры использования py-ai-desktop.

Демонстрирует:
1. Function Calling / Tools
2. ToolExecutor (автоматическое выполнение)
3. MCP клиент для подключения внешних инструментов
"""
import asyncio
import sys
import json

import httpx

from src.core import (
    # LLM
    LLMFactory, Message, LLMError,
    # Tools
    ToolRegistry, tool, CalculatorTool, CurrentTimeTool,
    # Executor
    ToolExecutor, ExecutorConfig,
    # Conversation
    Conversation, ConversationConfig,
    # MCP
    MCPClient, MCPToolAdapter,
    # Config
    Settings,
)


# ─── Примеры Tools ───

async def example_basic_tools():
    """
    Пример 1: Базовое использование инструментов.
    Определяем функции и регистрируем их как инструменты.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 1: Базовые Tools")
    print("="*60)
    
    settings = Settings()
    
    # Создаём реестр инструментов
    registry = ToolRegistry()
    
    # Регистрируем через декоратор
    @registry.register
    def get_weather(city: str) -> str:
        """
        Получить текущую погоду в городе.
        
        Args:
            city: Название города
        """
        # Симуляция (в реальности - вызов weather API)
        weather_data = {
            "moscow": "Облачно, +5°C",
            "london": "Дождь, +12°C",
            "tokyo": "Ясно, +18°C",
        }
        return weather_data.get(city.lower(), f"Погода в {city}: данные недоступны")
    
    # Регистрируем встроенный инструмент
    registry.register_tool(CalculatorTool())
    registry.register_tool(CurrentTimeTool())
    
    # Выводим схемы инструментов
    print("\nЗарегистрированные инструменты:")
    for schema in registry.get_schemas():
        func = schema["function"]
        print(f"  - {func['name']}: {func['description'][:50]}...")
    
    # Создаём LLM
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    # Создаём executor
    executor = ToolExecutor(
        llm=llm,
        tools=registry,
        config=ExecutorConfig(verbose=True),
    )
    
    # Запрос, который требует вызова инструментов
    messages = [
        Message(role="system", content="Ты полезный помощник. Используй доступные инструменты."),
        Message(role="user", content="Какая погода в Москве? И сколько будет 123 * 456?"),
    ]
    
    try:
        print("\nЗапрос: Какая погода в Москве? И сколько будет 123 * 456?")
        print("\nВыполнение:")
        
        response = await executor.run(messages)
        
        print(f"\nФинальный ответ:\n{response.content}")
        
    except LLMError as e:
        print(f"Ошибка: {e}")
    finally:
        await llm.close()


async def example_conversation_with_tools():
    """
    Пример 2: Диалог с инструментами.
    Многократное взаимодействие с сохранением контекста.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 2: Диалог с Tools")
    print("="*60)
    
    settings = Settings()
    
    # Создаём инструменты
    registry = ToolRegistry()
    
    @registry.register
    def search_docs(query: str) -> str:
        """
        Поиск в документации.
        
        Args:
            query: Поисковый запрос
        """
        docs = {
            "python": "Python — высокоуровневый язык программирования.",
            "async": "async/await используется для асинхронного программирования.",
            "list": "list — изменяемая последовательность в Python.",
        }
        for key, value in docs.items():
            if key in query.lower():
                return value
        return "Документация не найдена."
    
    @registry.register
    def get_current_time() -> str:
        """Получить текущее время."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    # Создаём диалог
    conversation = Conversation(config=ConversationConfig(
        system_prompt="Ты помощник по Python. Используй инструменты для поиска информации.",
    ))
    
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    executor = ToolExecutor(
        llm=llm,
        tools=registry,
        config=ExecutorConfig(verbose=True),
    )
    
    # Диалог
    questions = [
        "Что такое async в Python?",
        "А текущее время?",
        "Расскажи про списки.",
    ]
    
    try:
        for question in questions:
            print(f"\n--- Пользователь: {question} ---")
            
            conversation.add_user_message(question)
            
            response = await executor.run(conversation.get_messages())
            
            print(f"Ассистент: {response.content[:200]}...")     # type: ignore
            
            conversation.add_assistant_message(response.content) # type: ignore
            
    except LLMError as e:
        print(f"Ошибка: {e}")
    finally:
        await llm.close()


async def example_custom_tool_decorator():
    """
    Пример 3: Различные способы создания инструментов.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 3: Способы создания Tools")
    print("="*60)
    
    registry = ToolRegistry()
    
    # Способ 1: Декоратор @tool
    @tool
    def simple_function(x: int, y: int) -> int:
        """Сложить два числа."""
        return x + y
    
    registry.register_tool(simple_function) # type: ignore
    
    # Способ 2: Декоратор @registry.register
    @registry.register
    def another_function(text: str) -> str:
        """Преобразовать текст в верхний регистр."""
        return text.upper()
    
    # Способ 3: Явная регистрация с кастомным именем
    def my_function(data: str) -> str:
        """Обработать данные."""
        return f"Processed: {data}"
    
    registry.register_function(my_function, name="process_data")
    
    # Способ 4: Класс-инструмент
    class MyTool:
        name = "my_custom_tool"
        description = "Мой кастомный инструмент"
        parameters = {}
        
        async def execute(self, **kwargs):
            return "Custom tool executed!"
    
    # Выводим результат
    print("\nИнструменты в реестре:")
    for name in registry.get_tool_names():
        t = registry.get(name)
        print(f"  - {name}: {t.description[:50] if hasattr(t, 'description') else ''}") # type: ignore
    
    # Тестируем выполнение
    print("\nТест выполнения:")
    result = await registry.execute("simple_function", {"x": 5, "y": 3})
    print(f"  simple_function(5, 3) = {result}")
    
    result = await registry.execute("another_function", {"text": "hello"})
    print(f"  another_function('hello') = {result}")


async def example_mcp_client():
    """
    Пример 4: Подключение к MCP серверу.
    
    MCP (Model Context Protocol) — протокол для подключения
    внешних инструментов от MCP-совместимых серверов.
    
    Использует StreamableHTTP транспорт (новый стандарт).
    """
    print("\n" + "="*60)
    print("ПРИМЕР 4: MCP Клиент (StreamableHTTP)")
    print("="*60)
    
    # MCP сервер с StreamableHTTP транспортом
    # Можно передавать параметры прямо в URL
    mcp_server_url = "http://localhost:3000/mcp?tochka-bank-api-key=sandbox.jwt.token&is-test-env=true"
    
    print(f"\nПодключение к MCP серверу: {mcp_server_url}")
    
    try:
        async with MCPClient(mcp_server_url) as client:
            # Информация о сервере
            info = client.server_info
            print(f"\n✓ Подключено!")
            print(f"  Сервер: {info.name} v{info.version}")  # type: ignore
            print(f"  Protocol: {info.protocol_version}")    # type: ignore
            
            if client.session_id:
                print(f"  Session ID: {client.session_id}")
            
            # Возможности сервера
            caps = client.capabilities
            print(f"\n  Возможности:")
            print(f"    - Tools: {'✓' if caps.tools else '✗'}")          # type: ignore
            print(f"    - Resources: {'✓' if caps.resources else '✗'}")  # type: ignore
            print(f"    - Prompts: {'✓' if caps.prompts else '✗'}")      # type: ignore
            
            tools = client.tools
            print(f"\n  Инструменты ({len(tools)}):")
            for tool in tools:
                desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
                print(f"    - {tool.name}: {desc}")
            
            params = {"customerCode": "1234567ab",
                      "fromDate": "2025-01-01",
                      "toDate": "2026-03-20",
                      "format": "table"
                    }

            if tools:
                first_tool = "get_payments_summary"
                print(f"\n  Вызов инструмента '{first_tool}'...")
                result = await client.call_tool(first_tool, arguments=params)
                
                if result.is_error:
                    print(f"    ✗ Ошибка: {result.text}")
                else:
                    text = result.text or json.dumps(result.content, ensure_ascii=False)
                    print(f"    ✓ Результат:")
                    print(text)
            
    except httpx.ConnectError as e:
        print(f"\n✗ Не удалось подключиться к серверу")
        print(f"  Убедитесь, что MCP сервер запущен: {mcp_server_url}")
    except httpx.HTTPStatusError as e:
        print(f"\n✗ HTTP ошибка: {e.response.status_code}")
        print(f"  Response: {e.response.text[:200]}")
    except Exception as e:
        print(f"\n✗ Ошибка: {type(e).__name__}: {e}")


async def example_mcp_with_executor():
    """
    Пример 5: Использование MCP инструментов через ToolExecutor.
    Интеграция внешних MCP инструментов в цикл вызова LLM.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 5: MCP + ToolExecutor")
    print("="*60)
    
    settings = Settings()
    mcp_server_url = "http://localhost:3000/mcp?api-key=sandbox&is-test-env=true"
    
    print(f"\nПодключение к MCP серверу...")
    
    try:
        # Подключаемся к MCP серверу
        mcp_client = MCPClient(mcp_server_url)
        await mcp_client.connect()
        
        info = mcp_client.server_info
        print(f"✓ Сервер: {info.name} v{info.version}")  # type: ignore
        print(f"  Инструментов: {len(mcp_client.tools)}")
        
        # Создаём адаптер и реестр
        adapter = MCPToolAdapter(mcp_client)
        registry = adapter.create_registry()
        
        # Можно добавить и локальные инструменты
        registry.register_tool(CurrentTimeTool())
        
        # Создаём LLM и executor
        llm = LLMFactory.create(
            provider="deepseek",
            api_key=settings.deepseek_api_key,
        )
        
        executor = ToolExecutor(
            llm=llm,
            tools=registry,
            config=ExecutorConfig(verbose=True, max_iterations=5),
        )
        
        # Запрос
        messages = [
            Message(role="user", content="Используй доступные MCP инструменты."),
        ]
        
        response = await executor.run(messages)
        print(f"\nОтвет: {response.content}")
        
        await llm.close()
        await mcp_client.close()
        
    except httpx.ConnectError:
        print("✗ Не удалось подключиться к MCP серверу")
        print(f"  URL: {mcp_server_url}")
    except Exception as e:
        print(f"✗ Ошибка: {type(e).__name__}: {e}")


async def example_tool_choice():
    """
    Пример 6: Управление tool_choice.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 6: tool_choice")
    print("="*60)
    
    settings = Settings()
    
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    # Разные варианты tool_choice
    
    # 1. "auto" - LLM сама решает
    print("\n1. tool_choice='auto' (LLM решает сама):")
    response = await llm.chat(
        messages=[Message(role="user", content="Привет!")],
        tools=registry.get_schemas(),
        tool_choice="auto",
    )
    print(f"   Tool calls: {response.has_tool_calls}")
    
    # 2. "none" - запрет инструментов
    print("\n2. tool_choice='none' (запрет):")
    response = await llm.chat(
        messages=[Message(role="user", content="Сколько будет 2+2?")],
        tools=registry.get_schemas(),
        tool_choice="none",
    )
    print(f"   Tool calls: {response.has_tool_calls}")
    print(f"   Ответ: {response.content[:100]}...") # type: ignore
    
    await llm.close()


# ─── Main ───

async def main():
    """Запуск всех примеров."""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║      py-ai-desktop: Function Calling & MCP Examples        ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    examples = {
        "1": ("Базовые Tools", example_basic_tools),
        "2": ("Диалог с Tools", example_conversation_with_tools),
        "3": ("Способы создания Tools", example_custom_tool_decorator),
        "4": ("MCP Клиент", example_mcp_client),
        "5": ("MCP + Executor", example_mcp_with_executor),
        "6": ("tool_choice", example_tool_choice),
    }
    
    await example_mcp_client()
    # # Если передан аргумент — запускаем конкретный пример
    # if len(sys.argv) > 1:
    #     choice = sys.argv[1]
    #     if choice in examples:
    #         await examples[choice][1]()
    #         return
    
    # # Иначе запускаем все по очереди
    # for key, (name, func) in examples.items():
    #     try:
    #         await func()
    #     except Exception as e:
    #         print(f"\nОшибка в примере '{name}': {e}")
        
    #     # Пауза между примерами
    #     if key != list(examples.keys())[-1]:
    #         await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.run(main())
    print('\n✓ Готово!')
