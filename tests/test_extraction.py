"""M1.2 — selective extraction: transcript -> typed memory candidates.

The LLM is stubbed (canned responses): we test the pipeline's parsing, typing,
dedupe-threading and failure behavior, not the model's judgment (that was the S1
spike). Skip-KB selectivity is the model's call — here we only assert the
pipeline hands it the selectivity rules to act on.
"""

import pytest

from memora.extraction import (
    ExtractedMemory,
    ExtractionError,
    PriorMemory,
    extract_memories,
)
from memora.models import MemoryType
from memora.providers.base import LLMProvider


class StubLLM(LLMProvider):
    """Returns a canned reply and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, str | None]] = []

    async def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 4096
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply


_TWO_MEMORIES = """{"memories": [
  {"type": "preference", "content": "Prefers email over phone.", "supersedes": null, "confidence": 0.9},
  {"type": "entity_fact", "content": "On the Pro plan, annual billing.", "supersedes": null, "confidence": 0.8}
]}"""


async def test_parses_typed_memories() -> None:
    result = await extract_memories("Customer chat...", StubLLM(_TWO_MEMORIES))

    assert [m.type for m in result] == [MemoryType.preference, MemoryType.entity_fact]
    assert result[0].content == "Prefers email over phone."
    assert isinstance(result[0], ExtractedMemory)


async def test_tolerates_markdown_fenced_json() -> None:
    # S1 finding: the model wraps JSON in ```json fences despite instructions not to
    fenced = f"```json\n{_TWO_MEMORIES}\n```"
    result = await extract_memories("chat", StubLLM(fenced))

    assert len(result) == 2
    assert result[1].type is MemoryType.entity_fact


async def test_empty_extraction_returns_empty_list() -> None:
    result = await extract_memories("just small talk", StubLLM('{"memories": []}'))
    assert result == []


async def test_invalid_type_is_filtered_not_fatal() -> None:
    reply = """{"memories": [
      {"type": "made_up_type", "content": "junk"},
      {"type": "policy", "content": "Refunds within 30 days."}
    ]}"""
    result = await extract_memories("chat", StubLLM(reply))

    assert len(result) == 1
    assert result[0].type is MemoryType.policy


async def test_unparseable_response_raises_extraction_error() -> None:
    with pytest.raises(ExtractionError):
        await extract_memories("chat", StubLLM("Sorry, I could not help with that."))


async def test_existing_memories_are_threaded_and_supersedes_surfaced() -> None:
    stub = StubLLM(
        '{"memories": [{"type": "entity_fact", "content": "Email is new@x.com",'
        ' "supersedes": "mem_1", "confidence": 0.95}]}'
    )
    result = await extract_memories(
        "Customer: my email changed",
        stub,
        existing=[PriorMemory(id="mem_1", content="Email is old@x.com")],
    )

    # the replaced belief is surfaced for the write path to resolve
    assert result[0].supersedes == "mem_1"
    # dedupe works only if current state actually reaches the model
    sent = stub.calls[0]["prompt"] or ""
    assert "mem_1" in sent
    assert "old@x.com" in sent


async def test_sends_transcript_and_selectivity_rules_to_model() -> None:
    stub = StubLLM('{"memories": []}')
    await extract_memories("Customer: I only want email.", stub)

    assert "Customer: I only want email." in (stub.calls[0]["prompt"] or "")
    # the KB-exclusion rule is what makes skip-KB-content work
    assert "knowledge base" in (stub.calls[0]["system"] or "").lower()
