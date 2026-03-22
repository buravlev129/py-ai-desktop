"""
py-ai-desktop core module.

Экспортирует основные классы для работы с LLM.
"""

# Models
from .models import (
    Message,
    LLMResponse,
    LLMConfig,
    RetrySettings,
    StreamChunk,
    BaseLLMConnector,
    ToolCallData,
    FunctionCallData,
)

# Config
from .config import Settings

# Factory
from .llmfactory import LLMFactory

# Conversation
from .conversation import (
    Conversation,
    ConversationConfig,
    ConversationManager,
)

# Retry
from .retry import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ContextLengthError,
    ServerError,
    TimeoutError,
    NetworkError,
    with_retry,
    RetryConfig,
    RetryHandler,
)

# Tools
from .tools import (
    BaseTool,
    FunctionTool,
    ToolRegistry,
    ToolParameter,
    ToolSchema,
    ToolCall,
    ToolResult,
    tool,
    CalculatorTool,
    CurrentTimeTool,
)

# Executor
from .executor import (
    ToolExecutor,
    ExecutorConfig,
    create_tool_result_message,
    create_assistant_tool_call_message,
)

# MCP
from .mcp_client import (
    MCPClient,
    MCPTool,
    MCPToolResult,
    MCPServerInfo,
    MCPCapabilities,
    MCPClientConfig,
    MCPToolAdapter,
    MCPToolWrapper,
    TransportType,
)


__all__ = [
    # Models
    "Message",
    "LLMResponse",
    "LLMConfig",
    "RetrySettings",
    "StreamChunk",
    "BaseLLMConnector",
    "ToolCallData",
    "FunctionCallData",
    # Config
    "Settings",
    # Factory
    "LLMFactory",
    # Conversation
    "Conversation",
    "ConversationConfig",
    "ConversationManager",
    # Retry
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ContextLengthError",
    "ServerError",
    "TimeoutError",
    "NetworkError",
    "with_retry",
    "RetryConfig",
    "RetryHandler",
    # Tools
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "ToolParameter",
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "tool",
    "CalculatorTool",
    "CurrentTimeTool",
    # Executor
    "ToolExecutor",
    "ExecutorConfig",
    "create_tool_result_message",
    "create_assistant_tool_call_message",
    # MCP
    "MCPClient",
    "MCPTool",
    "MCPToolResult",
    "MCPServerInfo",
    "MCPCapabilities",
    "MCPClientConfig",
    "MCPToolAdapter",
    "MCPToolWrapper",
    "TransportType",
]