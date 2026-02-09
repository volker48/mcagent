import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def ls(path: str | None = None) -> str:
    cmd = ["ls"]
    if path:
        cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def read(path: str) -> str:
    with open(path, "r") as fp:
        return fp.read()


TOOLS: dict[str, Tool] = {
    "ls": Tool(
        name="ls",
        description="List the contents of the given path",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to list, e.g. /Users/marcusmccurdy/code/mcagent",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        fn=ls,
    ),
    "read": Tool(
        name="read",
        description="Read the contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read, e.g. /Users/jsmith/stuff.txt",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        fn=read,
    ),
}
