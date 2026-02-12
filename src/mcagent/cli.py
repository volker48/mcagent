import argparse
import os
from dataclasses import asdict, dataclass
from typing import cast

import httpx
from dotenv import load_dotenv

from mcagent.messages import (
    AnthropicRequest,
    AnthropicResponse,
    Message,
    Role,
    TextBlock,
    ToolResult,
    ToolUseBlock,
)
from mcagent.tools import TOOLS
from mcagent.types import Models

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY")

HEADERS: dict[str, str] = {
    "anthropic-version": "2023-06-01",
    "x-api-key": API_KEY,
}

HTTP_TIMEOUT = httpx.Timeout(connect=10.0, write=30.0, read=300.0, pool=10.0)

SYSTEM = """You are McAgent a helpful former Austrian movie start who is now an expert software engineer. 
You always respond in your iconic accent wiht clever one liners and copious emojis. 
You want to help but you also aren't afraid to push back just like you aren't afraid to push some weight."""


@dataclass(frozen=True)
class Args:
    model: Models
    max_tokens: int


def setup_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["claude-opus-4-6", "claude-haiku-4-5", "claude-sonnet-4-5"],
        default="claude-sonnet-4-5",
        help="Select one of the Claude models.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    ns = parser.parse_args()
    return Args(
        model=cast(Models, ns.model),
        max_tokens=ns.max_tokens,
    )


def handle_resp(resp: AnthropicResponse, conversation: list[Message]):
    tool_results = []
    for item in resp.content:
        conversation.append(Message(Role.ASSISTANT, content=[item]))
        match item:
            case TextBlock() as tb:
                print("Agent: " + tb.text)
            case ToolUseBlock(id=id, input=input, name=name):
                tool = TOOLS[name]
                tool_output = tool.fn(**input)
                tool_results.append(ToolResult(tool_use_id=id, content=tool_output))
    if tool_results:
        conversation.append(
            Message(
                role=Role.USER,
                content=tool_results,
            )
        )


def send(
    client: httpx.Client,
    max_tokens: int,
    model: Models,
    tools: list[dict],
    conversation=list[Message],
) -> AnthropicResponse:
    anth_req = AnthropicRequest(
        max_tokens=max_tokens,
        model=model,
        messages=conversation,
        tools=tools,
        system=SYSTEM,
    )
    resp = client.post("messages", json=asdict(anth_req))
    resp_body = resp.json()
    return AnthropicResponse.from_json(resp_body)


def main():
    args = setup_args()
    conversation: list[Message] = []
    tools = [tool.to_dict() for tool in TOOLS.values()]
    with httpx.Client(
        base_url="https://api.anthropic.com/v1/",
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
    ) as client:
        while True:
            usr_msg = input("User: ")
            conversation.append(Message(role=Role.USER, content=usr_msg))
            while True:
                resp = send(
                    client=client,
                    max_tokens=args.max_tokens,
                    model=args.model,
                    tools=tools,
                    conversation=conversation,
                )
                handle_resp(resp, conversation)
                if resp.stop_reason == "end_turn":
                    break


if __name__ == "__main__":
    main()
