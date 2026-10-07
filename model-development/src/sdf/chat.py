"""Minimal chat-prompt types and environment loading for SDF batch requests.

These replace the three ``safetytooling.data_models`` classes the SDF pipeline
used. They serialise identically, so ``Prompt.model_hash()`` (part of every
Azure batch ``custom_id``) and ``Prompt.openai_format()`` (the request body)
are byte-for-byte what the paper run produced.
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str

    def openai_format(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: Sequence[ChatMessage]

    def model_hash(self) -> str:
        return hashlib.sha1(self.model_dump_json().encode()).hexdigest()

    def openai_format(self) -> list[dict[str, str]]:
        return [message.openai_format() for message in self.messages]


def user_prompt(content: str) -> Prompt:
    return Prompt(messages=[ChatMessage(role=MessageRole.user, content=content)])


def load_environment() -> None:
    """Load Azure credentials from ``.env`` in the working directory, if present.

    The batch client reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and
    AZURE_OPENAI_API_VERSION; exporting them directly works equally well.
    """
    logging.basicConfig(level=logging.WARNING)
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if not load_dotenv(override=True):
        LOGGER.info("No .env file found; using the existing environment.")
