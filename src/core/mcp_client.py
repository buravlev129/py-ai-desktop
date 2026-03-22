"""
MCP (Model Context Protocol) клиент.

Поддерживает транспорты:
- StreamableHTTP (новый стандарт) - POST запросы с JSON-RPC
- SSE (Server-Sent Events) - для старых серверов

Usage:
    # StreamableHTTP (по умолчанию)
    client = MCPClient("http://localhost:3000/mcp")
    await client.connect()

    # С параметрами в URL
    client = MCPClient("http://localhost:3000/mcp?api-key=sandbox")

    tools = await client.list_tools()
    result = await client.call_tool("my_tool", {"arg": "value"})

    await client.close()
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List
from datetime import datetime

import httpx


# ─── Transport Types ───

class TransportType(Enum):
    """Тип транспорта MCP"""
    STREAMABLE_HTTP = "streamable_http"  # Новый стандарт (POST)
    SSE = "sse"  # Server-Sent Events (GET + POST)


# ─── MCP Data Models ───

@dataclass
class MCPTool:
    """Инструмент MCP сервера"""
    name: str
    description: str
    input_schema: dict  # JSON Schema

    def to_openai_schema(self) -> dict:
        """Преобразовать в формат OpenAI tools"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }


@dataclass
class MCPToolResult:
    """Результат вызова MCP инструмента"""
    content: list[dict]  # список content блоков
    is_error: bool = False

    @property
    def text(self) -> Optional[str]:
        """Получить текстовый контент"""
        for block in self.content:
            if block.get("type") == "text":
                return block.get("text")
        return None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "isError": self.is_error,
        }


@dataclass
class MCPServerInfo:
    """Информация о MCP сервере"""
    name: str
    version: str
    protocol_version: str = "2024-11-05"


@dataclass
class MCPCapabilities:
    """Возможности сервера"""
    tools: bool = False
    resources: bool = False
    prompts: bool = False
    logging: bool = False


@dataclass
class MCPClientConfig:
    """Конфигурация MCP клиента"""
    timeout: float = 30.0
    request_timeout: float = 120.0
    transport: TransportType = TransportType.STREAMABLE_HTTP
    headers: dict = field(default_factory=dict)


# ─── JSON-RPC Types ───

def generate_request_id() -> str:
    """Генерировать уникальный ID запроса"""
    return str(uuid.uuid4())


def create_request(method: str, params: dict = None, request_id: str = None) -> dict: # type: ignore
    """Создать JSON-RPC 2.0 запрос"""
    request = {
        "jsonrpc": "2.0",
        "id": request_id or generate_request_id(),
        "method": method,
    }
    if params is not None:
        request["params"] = params   # type: ignore
    return request


def create_notification(method: str, params: dict = None) -> dict:  # type: ignore
    """Создать JSON-RPC 2.0 notification (без id)"""
    notification = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params  # type: ignore
    return notification


# ─── MCP Client ───

class MCPClient:
    """
    Клиент для подключения к MCP серверам.

    MCP (Model Context Protocol) - протокол от Anthropic
    для стандартизированного взаимодействия с инструментами.

    Supported Transports:
    - StreamableHTTP: POST запросы на один endpoint
    - SSE: Server-Sent Events (для старых серверов)

    Usage:
        async with MCPClient("http://localhost:3000/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
    """

    def __init__(
        self,
        server_url: str,
        config: Optional[MCPClientConfig] = None,
    ):
        """
        Args:
            server_url: URL MCP сервера
                StreamableHTTP: http://localhost:3000/mcp
                SSE: http://localhost:3000/sse
            config: Конфигурация клиента
        """
        self.server_url = server_url
        self.config = config or MCPClientConfig()

        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self._server_info: Optional[MCPServerInfo] = None
        self._capabilities: Optional[MCPCapabilities] = None
        self._tools: List[MCPTool] = []
        self._session_id: Optional[str] = None

    async def connect(self) -> MCPServerInfo:
        """
        Подключиться к MCP серверу и выполнить handshake.

        Returns:
            Информация о сервере
        """
        if self._connected:
            return self._server_info  # type: ignore

        # Создаём HTTP клиент
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.timeout,
                read=self.config.request_timeout,
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            headers=self.config.headers,
        )

        # Отправляем initialize request
        self._server_info, self._capabilities = await self._initialize()

        # Отправляем initialized notification
        await self._send_notification("notifications/initialized")

        # Загружаем инструменты если поддерживаются
        if self._capabilities and self._capabilities.tools:
            await self._load_tools()

        self._connected = True
        return self._server_info

    async def _initialize(self) -> tuple[MCPServerInfo, MCPCapabilities]:
        """Отправить initialize request и получить информацию о сервере"""

        request = create_request(
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "clientInfo": {
                    "name": "py-ai-desktop",
                    "version": "1.0.0",
                }
            }
        )

        response_data = await self._send_request_raw(request)

        if "error" in response_data:
            raise Exception(f"Initialize failed: {response_data['error']}")

        result = response_data.get("result", {})

        # Парсим capabilities
        capabilities_data = result.get("capabilities", {})
        capabilities = MCPCapabilities(
            tools="tools" in capabilities_data,
            resources="resources" in capabilities_data,
            prompts="prompts" in capabilities_data,
            logging="logging" in capabilities_data,
        )

        # Парсим server info
        server_info = result.get("serverInfo", {})

        return MCPServerInfo(
            name=server_info.get("name", "unknown"),
            version=server_info.get("version", "0.0.0"),
            protocol_version=result.get("protocolVersion", "2024-11-05"),
        ), capabilities

    async def _load_tools(self):
        """Загрузить список инструментов с сервера"""
        request = create_request(method="tools/list")
        response_data = await self._send_request_raw(request)

        if "error" in response_data:
            raise Exception(f"List tools failed: {response_data['error']}")

        result = response_data.get("result", {})
        self._tools = []

        for tool_data in result.get("tools", []):
            self._tools.append(MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
            ))

    async def _send_request_raw(self, request: dict) -> dict:
        """
        Отправить JSON-RPC запрос и получить сырой ответ.

        Для StreamableHTTP:
        - POST запрос на server_url
        - Content-Type: application/json
        - Ответ может быть JSON или SSE-stream

        Returns:
            Распарсенный JSON-RPC ответ
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        # Добавляем session ID если есть
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        response = await self._client.post(  # type: ignore
            self.server_url,
            json=request,
            headers=headers,
        )

        # Сохраняем session ID из ответа
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # Обрабатываем ответ в зависимости от типа
        if "text/event-stream" in content_type:
            # SSE ответ - парсим события
            return await self._parse_sse_response(response)
        else:
            # Обычный JSON ответ
            return response.json()

    async def _parse_sse_response(self, response: httpx.Response) -> dict:
        """
        Парсит SSE ответ и возвращает JSON-RPC response.

        SSE формат:
        event: message
        data: {"jsonrpc":"2.0", "id":"...", "result":{...}}
        """
        text = response.text
        lines = text.strip().split("\n")

        current_event = None
        data_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "" and data_lines:
                # Конец события
                data_str = "".join(data_lines)
                try:
                    data = json.loads(data_str)
                    # Возвращаем первый валидный JSON-RPC response
                    if "jsonrpc" in data:
                        return data
                except json.JSONDecodeError:
                    pass
                data_lines = []

        # Если не нашли валидный response, пробуем последний data
        if data_lines:
            try:
                return json.loads("".join(data_lines))
            except json.JSONDecodeError:
                pass

        raise Exception("Failed to parse SSE response")

    async def _send_notification(self, method: str, params: dict = None):  # type: ignore
        """Отправить JSON-RPC notification (без id)"""
        notification = create_notification(method, params)

        headers = {
            "Content-Type": "application/json",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        await self._client.post(  # type: ignore
            self.server_url,
            json=notification,
            headers=headers,
        )

    # ─── Public API ───

    @property
    def tools(self) -> List[MCPTool]:
        """Список доступных инструментов"""
        return self._tools

    @property
    def server_info(self) -> Optional[MCPServerInfo]:
        """Информация о сервере"""
        return self._server_info

    @property
    def capabilities(self) -> Optional[MCPCapabilities]:
        """Возможности сервера"""
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        """Проверить, подключен ли клиент"""
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        """ID сессии (если сервер его предоставил)"""
        return self._session_id

    async def list_tools(self) -> List[MCPTool]:
        """
        Получить список инструментов с сервера.

        Returns:
            Список MCPTool объектов
        """
        if not self._connected:
            await self.connect()
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict = None,  # type: ignore
    ) -> MCPToolResult:
        """
        Вызвать инструмент на MCP сервере.

        Args:
            name: Имя инструмента
            arguments: Аргументы для инструмента

        Returns:
            Результат выполнения
        """
        if not self._connected:
            await self.connect()

        request = create_request(
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments or {},
            }
        )

        response_data = await self._send_request_raw(request)

        if "error" in response_data:
            error = response_data["error"]
            return MCPToolResult(
                content=[{
                    "type": "text",
                    "text": error.get("message", str(error))
                }],
                is_error=True,
            )

        result = response_data.get("result", {})
        return MCPToolResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    def get_tool_schemas(self) -> List[dict]:
        """
        Получить схемы инструментов в формате OpenAI.

        Returns:
            Список схем для использования с LLM
        """
        return [tool.to_openai_schema() for tool in self._tools]

    async def close(self):
        """Закрыть соединение"""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._session_id = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()


# ─── MCP Tool Adapter ───

class MCPToolAdapter:
    """
    Адаптер для использования MCP инструментов с ToolRegistry.

    Позволяет интегрировать инструменты MCP сервера
    в локальный ToolRegistry для использования с ToolExecutor.

    Usage:
        mcp_client = MCPClient("http://localhost:3000/mcp")
        await mcp_client.connect()

        adapter = MCPToolAdapter(mcp_client)
        registry = adapter.create_registry()

        # Теперь можно использовать с ToolExecutor
        executor = ToolExecutor(llm, registry)
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    def create_mcp_tool(self, mcp_tool: MCPTool) -> "MCPToolWrapper":
        """Создать обёртку для MCP инструмента"""
        return MCPToolWrapper(
            mcp_client=self.mcp_client,
            mcp_tool=mcp_tool,
        )

    def create_registry(self) -> "ToolRegistry":   # type: ignore
        """Создать ToolRegistry с MCP инструментами"""
        from .tools import ToolRegistry

        registry = ToolRegistry()

        for mcp_tool in self.mcp_client.tools:
            wrapper = self.create_mcp_tool(mcp_tool)
            registry.register_tool(wrapper)   # type: ignore

        return registry


class MCPToolWrapper:
    """
    Обёртка над MCP инструментом для использования с ToolRegistry.
    """

    name: str
    description: str
    parameters: dict

    def __init__(self, mcp_client: MCPClient, mcp_tool: MCPTool):
        self._mcp_client = mcp_client
        self._mcp_tool = mcp_tool

        self.name = mcp_tool.name
        self.description = mcp_tool.description

        # Преобразуем inputSchema в параметры
        self.parameters = {}
        schema = mcp_tool.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        from .tools import ToolParameter

        for param_name, param_schema in properties.items():
            self.parameters[param_name] = ToolParameter(
                type=param_schema.get("type", "string"),
                description=param_schema.get("description", ""),
                enum=param_schema.get("enum"),
                default=None if param_name in required else None,
            )

    async def execute(self, **kwargs) -> Any:
        """Выполнить MCP инструмент"""
        result = await self._mcp_client.call_tool(self.name, kwargs)

        if result.is_error:
            raise Exception(result.text or "MCP tool error")

        return result.text or result.content

    def to_openai_schema(self) -> dict:
        """Схема для OpenAI"""
        return self._mcp_tool.to_openai_schema()
