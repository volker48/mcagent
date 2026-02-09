from typing import Literal

Models = Literal["claude-opus-4-6", "claude-haiku-4-5", "claude-sonnet-4-5"]
StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]
