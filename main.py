import asyncio
from src.core.llmfactory import LLMFactory
from src.core.models import Message
from src.core.config import Settings


async def main():

    settings = Settings()

    deepseek = LLMFactory.create(
        provider="deepseek",
        # model="deepseek-chat",
        api_key=settings.deepseek_api_key,
    )

    qwen = LLMFactory.create(
        provider="qwen",
        # model="qwen-plus",
        api_key=settings.qwen_api_key,
    )

    # Claude — автоматически используется AnthropicConnector
    claude = LLMFactory.create(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-...",
    )

    # Grok — тот же OpenAI-совместимый коннектор
    grok = LLMFactory.create(
        provider="grok",
        model="grok-2-latest",
        api_key="xai-...",
    )

    messages = [
        Message(role="system", content="Ты полезный помощник."),
        Message(role="user", content="Привет! Как дела?"),
    ]

    llm = [deepseek, qwen, claude, grok][0]
    try:
        response = await llm.chat(messages)
        print(f"{response.model}: {response.content[:100]}")

    except Exception as ex:
        print(str(ex))
    finally:
        await llm.close()

if __name__ == '__main__':

    asyncio.run(main())
    print('---')
