from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Self

from mcagent.types import Models, StopReason


@dataclass
class Block:
    @classmethod
    def from_json(cls, data: dict) -> "Block":
        match data:
            case {"type": "text"}:
                return TextBlock(**data)
            case {"type": "thinking"}:
                return ThinkingBlock(**data)
            case {"type": "tool_use"}:
                return ToolUseBlock(**data)
            case _:
                raise ValueError(f"Unknown block type: {data.get('type')}")


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class TextBlock(Block):
    text: str
    type: Literal["text"]


@dataclass
class ThinkingBlock(Block):
    signature: str
    thinking: str
    type: Literal["thinking"]


@dataclass
class ToolUseBlock(Block):
    id: str
    input: dict
    name: str
    type: Literal["tool_use"]


@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    type: Literal["tool_result"] = "tool_result"


@dataclass
class Message:
    role: Role
    content: str | list[ToolResult | Block]


@dataclass
class AnthropicRequest:
    max_tokens: int
    messages: list[Message]
    model: Models
    tools: list[dict] = field(default_factory=list)
    system: str = ""


@dataclass
class AnthropicResponse:
    id: str
    content: list[Block]
    stop_reason: StopReason

    @classmethod
    def from_json(cls, data: dict) -> Self:
        content = [Block.from_json(block) for block in data["content"]]
        return cls(id=data["id"], content=content, stop_reason=data["stop_reason"])
