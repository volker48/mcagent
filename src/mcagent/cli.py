import argparse
import os
from dataclasses import asdict, dataclass

import httpx
from dotenv import load_dotenv

from mcagent.messages import (
    AnthropicRequest,
    AnthropicResponse,
    Message,
    Role,
    ToolResult,
)
from mcagent.tools import TOOLS
from mcagent.types import Models

load_dotenv()


HEADERS = {
    "anthropic-version": "2023-06-01",
    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
}


@dataclass
class CliArgs:
    model: Models


def setup_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main():
    cli_args = setup_args()
    with httpx.Client(
        base_url="https://api.anthropic.com/v1/", headers=HEADERS
    ) as client:
        if cli_args.content:
            message = Message(role=Role.USER, content=cli_args.content)
            anth_req = AnthropicRequest(
                max_tokens=cli_args.max_tokens,
                model=cli_args.model,
                messages=[message],
            )
            resp = client.post("messages", json=asdict(anth_req))
            resp_body = resp.json()
            resp_parsed = AnthropicResponse.from_json(resp_body)
            print(resp_parsed)
        else:
            messages = []
            usr_msg = input("User: ")
            messages.append(Message(role=Role.USER, content=usr_msg))
            while True:
                anth_req = AnthropicRequest(
                    max_tokens=cli_args.max_tokens,
                    model=cli_args.model,
                    messages=messages,
                    tools=[tool.to_dict() for tool in TOOLS.values()],
                )
                resp = client.post("messages", json=asdict(anth_req))
                resp_body = resp.json()
                resp_parsed = AnthropicResponse.from_json(resp_body)
                match resp_parsed.stop_reason:
                    case "end_turn":
                        print("Agent: " + resp_parsed.content[0].text)
                        next_msg = input("User: ")
                        messages.append(Message(role=Role.USER, content=next_msg))
                    case "tool_use":
                        messages += [
                            Message(role=Role.ASSISTANT, content=[item])
                            for item in resp_parsed.content
                        ]
                        agent_resp = resp_parsed.content[0].text
                        print("Agent: " + agent_resp)
                        tool_name = resp_parsed.content[1].name
                        tool_args = resp_parsed.content[1].input
                        tool_use_id = resp_parsed.content[1].id
                        tool = TOOLS[tool_name]
                        result = tool.fn(**tool_args)
                        messages.append(
                            Message(
                                role=Role.USER,
                                content=[
                                    ToolResult(
                                        tool_use_id, type="tool_result", content=result
                                    )
                                ],
                            )
                        )


if __name__ == "__main__":
    main()
