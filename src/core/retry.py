"""
Retry механизмы и обработка ошибок для LLM API.
"""
from __future__ import annotations

import asyncio
import logging
import random
from functools import wraps
from typing import Callable, Type, Tuple, Any, Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# ─── Исключения ───

class LLMError(Exception):
    """Базовое исключение для LLM ошибок"""
    pass


class RateLimitError(LLMError):
    """Превышен лимит запросов (429)"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Ошибка аутентификации (401, 403)"""
    pass


class ModelNotFoundError(LLMError):
    """Модель не найдена"""
    pass


class ContextLengthError(LLMError):
    """Превышен лимит контекста"""
    pass


class ServerError(LLMError):
    """Ошибка сервера (5xx)"""
    pass


class TimeoutError(LLMError):
    """Таймаут запроса"""
    pass


class NetworkError(LLMError):
    """Сетевая ошибка"""
    pass


# ─── Парсинг ошибок ───

def parse_http_error(error: httpx.HTTPStatusError) -> LLMError:
    """
    Преобразует HTTP ошибку в специфичное LLM исключение.
    """
    status_code = error.response.status_code
    
    # Пытаемся извлечь детали из ответа
    try:
        error_data = error.response.json()
        error_message = error_data.get("error", {}).get("message", str(error))
    except:
        error_message = str(error)
    
    if status_code == 401:
        return AuthenticationError(f"Authentication failed: {error_message}")
    
    if status_code == 403:
        return AuthenticationError(f"Access forbidden: {error_message}")
    
    if status_code == 404:
        return ModelNotFoundError(f"Model or endpoint not found: {error_message}")
    
    if status_code == 429:
        # Пытаемся извлечь retry-after
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                retry_after = int(retry_after)
            except ValueError:
                retry_after = None
        return RateLimitError(f"Rate limit exceeded: {error_message}", retry_after=retry_after)
    
    if status_code == 400:
        # Проверяем на context length
        if "context" in error_message.lower() or "token" in error_message.lower():
            return ContextLengthError(f"Context length exceeded: {error_message}")
        return LLMError(f"Bad request: {error_message}")
    
    if 500 <= status_code < 600:
        return ServerError(f"Server error ({status_code}): {error_message}")
    
    return LLMError(f"HTTP error ({status_code}): {error_message}")


# ─── Retry декоратор ───

@dataclass
class RetryConfig:
    """Конфигурация retry стратегии"""
    max_retries: int = 3
    base_delay: float = 1.0  # базовая задержка в секундах
    max_delay: float = 60.0  # максимальная задержка
    exponential_base: float = 2.0  # база для экспоненциального роста
    jitter: bool = True  # добавлять случайность к задержке
    
    # Исключения, для которых нужен retry
    retry_exceptions: Tuple[Type[Exception], ...] = (
        RateLimitError,
        ServerError,
        TimeoutError,
        NetworkError,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Декоратор для повторных попыток с экспоненциальным откатом.
    
    Args:
        max_retries: Максимальное количество повторных попыток
        base_delay: Базовая задержка в секундах
        max_delay: Максимальная задержка в секундах
        exponential_base: База для экспоненциального роста задержки
        jitter: Добавлять ли случайность к задержке
        retry_exceptions: Кортеж исключений для retry
    
    Usage:
        @with_retry(max_retries=3)
        async def call_api():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions or RetryConfig.retry_exceptions,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except httpx.HTTPStatusError as e:
                    # Преобразуем HTTP ошибку в LLM ошибку
                    llm_error = parse_http_error(e)
                    
                    # Для RateLimitError проверяем retry_after
                    if isinstance(llm_error, RateLimitError):
                        if llm_error.retry_after:
                            delay = min(llm_error.retry_after, config.max_delay)
                        else:
                            delay = _calculate_delay(
                                attempt, config.base_delay, 
                                config.exponential_base, config.max_delay, config.jitter
                            )
                        
                        logger.warning(
                            f"Rate limit hit, waiting {delay}s before retry "
                            f"(attempt {attempt + 1}/{config.max_retries})"
                        )
                        
                        if attempt < config.max_retries:
                            await asyncio.sleep(delay)
                            continue
                    
                    # Для других retry-исключений
                    if isinstance(llm_error, config.retry_exceptions):
                        last_exception = llm_error
                        
                        if attempt < config.max_retries:
                            delay = _calculate_delay(
                                attempt, config.base_delay,
                                config.exponential_base, config.max_delay, config.jitter
                            )
                            logger.warning(
                                f"{llm_error.__class__.__name__}, retrying in {delay}s "
                                f"(attempt {attempt + 1}/{config.max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                    
                    # Не retry-исключение — пробрасываем сразу
                    raise llm_error
                
                except config.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = _calculate_delay(
                            attempt, config.base_delay,
                            config.exponential_base, config.max_delay, config.jitter
                        )
                        logger.warning(
                            f"{e.__class__.__name__}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{config.max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
            
            # Все попытки исчерпаны
            if last_exception:
                raise last_exception
            
            # Этого не должно произойти
            raise RuntimeError("Retry logic error")
        
        return wrapper
    
    return decorator


def _calculate_delay(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter: bool
) -> float:
    """
    Вычисляет задержку для текущей попытки.
    Использует экспоненциальный отступ с опциональным jitter.
    """
    delay = base_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)
    
    if jitter:
        # Добавляем случайность от 50% до 100% от задержки
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


# ─── Класс для retry с состоянием ───

class RetryHandler:
    """
    Класс для обработки retry с возможностью настройки через конфиг.
    Может использоваться как обёртка над функциями.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Выполнить функцию с retry логикой"""
        
        @with_retry(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            exponential_base=self.config.exponential_base,
            jitter=self.config.jitter,
            retry_exceptions=self.config.retry_exceptions,
        )
        async def wrapped():
            return await func(*args, **kwargs)
        
        return await wrapped()

