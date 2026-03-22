"""
Примеры использования py-ai-desktop.

Демонстрирует:
1. Обычный запрос к LLM
2. Streaming (потоковые ответы)
3. Conversation Manager (управление диалогом)
4. Retry и обработка ошибок
"""
import asyncio
import sys

from src.core.llmfactory import LLMFactory
from src.core.models import Message, RetrySettings
from src.core.config import Settings
from src.core.conversation import Conversation, ConversationConfig
from src.core.retry import LLMError, RateLimitError


async def example_basic_chat():
    """
    Пример 1: Базовый запрос к LLM.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 1: Базовый запрос")
    print("="*60)
    
    settings = Settings()
    
    # Создаём клиент через фабрику
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    messages = [
        Message(role="system", content="Ты полезный помощник."),
        Message(role="user", content="Напиши короткое стихотворение о программировании."),
    ]
    
    try:
        response = await llm.chat(messages)
        print(f"Модель: {response.model}")
        print(f"Ответ:\n{response.content}")
        if response.usage:
            print(f"Токены: {response.usage}")
    except LLMError as e:
        print(f"Ошибка LLM: {e}")
    finally:
        await llm.close()


async def example_streaming():
    """
    Пример 2: Streaming (потоковые ответы).
    Ответ выводится по мере генерации.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 2: Streaming (потоковые ответы)")
    print("="*60)
    
    settings = Settings()
    
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    messages = [
        Message(role="system", content="Ты полезный помощник."),
        Message(role="user", content="Расскажи короткую историю о роботе, который научился мечтать."),
    ]
    
    print("Ответ (потоковый вывод):\n")
    print("-" * 40)
    
    try:
        async for chunk in llm.chat_stream(messages): # type: ignore
            # Печатаем каждый чанк без перевода строки
            print(chunk.content, end="", flush=True)
        print()  # финальный перевод строки
    except LLMError as e:
        print(f"\nОшибка: {e}")
    finally:
        await llm.close()


async def example_conversation():
    """
    Пример 3: Conversation Manager.
    Управление историей диалога с сохранением контекста.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 3: Conversation Manager")
    print("="*60)
    
    settings = Settings()
    
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
    )
    
    # Создаём диалог с кастомным system prompt
    config = ConversationConfig(
        system_prompt="Ты краткий и точный помощник. Отвечай не более 2 предложений.",
        max_messages=10,
    )
    conversation = Conversation(config=config)
    
    # Несколько вопросов в рамках одного диалога
    questions = [
        "Что такое Python?",
        "А какие у него основные преимущества?",
        "Как установить пакет через pip?",
    ]
    
    try:
        for i, question in enumerate(questions, 1):
            print(f"\n--- Вопрос {i} ---")
            print(f"Пользователь: {question}")
            
            # Добавляем вопрос в историю
            conversation.add_user_message(question)
            
            # Отправляем весь контекст
            response = await llm.chat(conversation.get_messages())
            
            print(f"Ассистент: {response.content}")
            
            # Добавляем ответ в историю
            conversation.add_assistant_message(response.content)     # type: ignore
        
        # Статистика диалога
        print("\n--- Статистика диалога ---")
        stats = conversation.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except LLMError as e:
        print(f"Ошибка: {e}")
    finally:
        await llm.close()


async def example_retry():
    """
    Пример 4: Retry и обработка ошибок.
    Автоматические повторные попытки при ошибках.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 4: Retry и обработка ошибок")
    print("="*60)
    
    settings = Settings()
    
    # Настраиваем retry
    retry_settings = RetrySettings(
        enabled=True,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
    )
    
    llm = LLMFactory.create(
        provider="deepseek",
        api_key=settings.deepseek_api_key,
        retry=retry_settings,  # передаём настройки retry
    )
    
    messages = [
        Message(role="user", content="Привет!"),
    ]
    
    try:
        print("Отправляем запрос (с автоматическим retry при ошибках)...")
        response = await llm.chat(messages)
        print(f"Ответ: {response.content[:100]}...")     # type: ignore
        print("Успешно!")
        
    except RateLimitError as e:
        print(f"Превышен лимит запросов. Попробуйте позже.")
        if e.retry_after:
            print(f"Рекомендуемое время ожидания: {e.retry_after} сек")
            
    except LLMError as e:
        print(f"Ошибка LLM: {e}")
        
    finally:
        await llm.close()


async def example_save_conversation():
    """
    Пример 5: Сохранение и загрузка диалога.
    """
    print("\n" + "="*60)
    print("ПРИМЕР 5: Сохранение/загрузка диалога")
    print("="*60)
    
    # Создаём диалог
    conversation = Conversation(
        config=ConversationConfig(system_prompt="Ты помощник.")
    )
    
    # Добавляем сообщения
    conversation.add_user_message("Привет!")
    conversation.add_assistant_message("Привет! Чем могу помочь?")
    conversation.add_user_message("Как дела?")
    
    # Сериализуем в JSON
    json_str = conversation.to_json()
    print("Сериализованный диалог:")
    print(json_str[:300] + "...")
    
    # Десериализуем
    loaded = Conversation.from_json(json_str)
    print(f"\nЗагружено сообщений: {loaded.message_count}")
    print(f"Последнее сообщение: {loaded.get_last_message().content}") # type: ignore


async def main():
    """Запуск всех примеров."""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║           py-ai-desktop: Примеры использования             ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Меню выбора примера
    examples = {
        "1": ("Базовый запрос", example_basic_chat),
        "2": ("Streaming", example_streaming),
        "3": ("Conversation Manager", example_conversation),
        "4": ("Retry", example_retry),
        "5": ("Сохранение диалога", example_save_conversation),
    }
    
    # Если передан аргумент — запускаем конкретный пример
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            await examples[choice][1]()
            return
    
    # Иначе запускаем все по очереди
    for key, (name, func) in examples.items():
        try:
            await func()
        except Exception as e:
            print(f"\nОшибка в примере '{name}': {e}")
        
        # Пауза между примерами
        if key != list(examples.keys())[-1]:
            await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.run(main())
    print('\nГотово!')

