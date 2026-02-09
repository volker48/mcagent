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

SYSTEM = """You are McAgent a helpful former Austrian movie start who is now an expert software engineer. 
You always respond with an Austrian flair and copious emojis and clever one liners and your signature accent. 
You want to help but you also aren't afraid to push back just like you aren't afraid to push some weight."""


@dataclass(frozen=True)
class Args:
    content: str | None
    model: Models
    max_tokens: int


def setup_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("content", nargs="?", default=None)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["claude-opus-4-6", "claude-haiku-4-5", "claude-sonnet-4-5"],
        default="claude-haiku-4-5",
        help="Select one of the Claude models.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    ns = parser.parse_args()
    return Args(
        content=ns.content,
        model=cast(Models, ns.model),
        max_tokens=ns.max_tokens,
    )


def handle_resp(resp: AnthropicResponse, conversation: list[Message]):
    for item in resp.content:
        conversation.append(Message(Role.ASSISTANT, content=[item]))
        match item:
            case TextBlock() as tb:
                print("Agent: " + tb.text)
    match resp.stop_reason:
        case "end_turn":
            next_msg = input("User: ")
            conversation.append(Message(role=Role.USER, content=next_msg))
        case "tool_use":
            match resp.content:
                case [
                    *_,
                    ToolUseBlock(name=tool_name, input=tool_args, id=tool_use_id),
                ]:
                    tool = TOOLS[tool_name]
                    result = tool.fn(**tool_args)
                    conversation.append(
                        Message(
                            role=Role.USER,
                            content=[
                                ToolResult(
                                    tool_use_id,
                                    type="tool_result",
                                    content=result,
                                )
                            ],
                        )
                    )


def single_message(client: httpx.Client, cli_args: Args):
    if not cli_args.content:
        raise ValueError("No content")
    conversation = [Message(role=Role.USER, content=cli_args.content)]
    anth_req = AnthropicRequest(
        max_tokens=cli_args.max_tokens,
        model=cli_args.model,
        messages=conversation,
        tools=[tool.to_dict() for tool in TOOLS.values()],
        system=SYSTEM,
    )
    resp = client.post("messages", json=asdict(anth_req))
    resp_body = resp.json()
    resp_parsed = AnthropicResponse.from_json(resp_body)
    handle_resp(resp_parsed, conversation=conversation)


def main():
    cli_args = setup_args()
    with httpx.Client(
        base_url="https://api.anthropic.com/v1/", headers=HEADERS
    ) as client:
        if cli_args.content:
            single_message(client, cli_args)
        else:
            conversation = []
            usr_msg = input("User: ")
            conversation.append(Message(role=Role.USER, content=usr_msg))
            while True:
                anth_req = AnthropicRequest(
                    max_tokens=cli_args.max_tokens,
                    model=cli_args.model,
                    messages=conversation,
                    tools=[tool.to_dict() for tool in TOOLS.values()],
                    system=SYSTEM,
                )
                resp = client.post("messages", json=asdict(anth_req))
                resp_body = resp.json()
                resp_parsed = AnthropicResponse.from_json(resp_body)
                handle_resp(resp_parsed, conversation)


if __name__ == "__main__":
    main()
