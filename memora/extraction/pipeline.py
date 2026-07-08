"""Selective extraction — a conversation transcript becomes typed memory candidates.

The prompt and selectivity rules are ported from the S1 spike (validated on real
support transcripts: precision ~0.88, zero KB leakage, supersession 5/5). Two S1
findings are baked in here: the model returns JSON in markdown fences despite
instructions (so parsing is fence-tolerant), and `correction` typing is the weak
spot (so the prompt carries a correction few-shot).

Extraction runs behind the async worker (M1.3); it does not touch the store — it
just turns text into candidates. Scope/actor attribution is threaded in later (M1.4).
"""

import json

import structlog
from pydantic import BaseModel, ValidationError

from memora.models import MemoryType
from memora.providers.base import LLMProvider

log = structlog.get_logger()


class ExtractionError(Exception):
    """The model response could not be parsed into memory candidates (worker may retry)."""


class PriorMemory(BaseModel):
    """An already-stored memory passed in so the model can dedupe/supersede against it."""

    id: str
    content: str


class ExtractedMemory(BaseModel):
    """One memory candidate the model pulled from a transcript.

    `type` reuses the M1.1 taxonomy — an off-vocabulary type fails validation and the
    candidate is dropped. `supersedes` is the id (or description) of a belief this
    replaces; the write path resolves it into `superseded_by`.
    """

    type: MemoryType
    content: str
    supersedes: str | None = None
    confidence: float | None = None


_SYSTEM_PROMPT = """You are the memory-extraction step of an AI-agent memory engine. \
From the conversation transcript, extract ONLY durable, reusable memories worth recalling in \
future conversations with this customer.

Memory types:
- entity_fact — a stable attribute of the customer/account (plan, address, email, billing \
cycle, environment such as "uses a Zscaler proxy")
- preference — how the customer wants to be treated (channel, tone, timing)
- correction — a human explicitly corrected a wrong belief the agent/system held
- commitment — a promise someone made that must be honored later
- policy — a business rule stated as applying generally, not just to this customer

Selectivity rules (critical):
- Do NOT extract troubleshooting steps, knowledge-base/help-article content, UI navigation \
paths, or generic product facts ("data export is on all paid plans"). Those live in the \
knowledge base, not memory.
- Do NOT extract small talk or transient conversation state.
- If the transcript states a value and then replaces it (the customer corrects themselves, a \
plan changes), extract ONLY the final value and record what it superseded.
- If EXISTING MEMORIES are provided and a new memory contradicts one, set "supersedes" to \
that memory's id.

Correction example (an implicit in-transcript correction):
  Agent: "I see your renewal is March 1." Customer: "No — we moved it to April 15 after the \
outage credit."
  -> {"type": "correction", "content": "Renewal date is April 15 (moved after an outage \
credit).", "supersedes": "renewal date March 1", "confidence": 0.95}

Return STRICT JSON only, of the form:
{"memories": [{"type": "<one type above>", "content": "<one self-contained sentence>", \
"supersedes": "<existing memory id, a short description of the replaced belief, or null>", \
"confidence": <0.0-1.0>}]}
If there is nothing worth remembering, return {"memories": []}."""


def _user_prompt(conversation: str, existing: list[PriorMemory] | None) -> str:
    parts = [f"TRANSCRIPT:\n{conversation}"]
    if existing:
        rendered = json.dumps([m.model_dump() for m in existing], indent=2)
        parts.append(f"EXISTING MEMORIES:\n{rendered}")
    return "\n\n".join(parts)


def _load_memories(raw: str) -> list[object]:
    """Pull the ``memories`` array out of a possibly fence-wrapped model response."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionError("no JSON object found in model response")
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError("model response was not valid JSON") from exc
    if not isinstance(obj, dict) or "memories" not in obj:
        raise ExtractionError("model response has no 'memories' array")
    memories = obj["memories"]
    if not isinstance(memories, list):
        raise ExtractionError("'memories' is not a list")
    return memories


async def extract_memories(
    conversation: str,
    llm: LLMProvider,
    *,
    existing: list[PriorMemory] | None = None,
) -> list[ExtractedMemory]:
    """Extract typed memory candidates from a transcript.

    Pass ``existing`` (current memories for this scope) to let the model dedupe and
    supersede against them. Raises ``ExtractionError`` if the response is unusable;
    individual malformed candidates are dropped, not fatal.
    """
    raw = await llm.generate(_user_prompt(conversation, existing), system=_SYSTEM_PROMPT)

    result: list[ExtractedMemory] = []
    for item in _load_memories(raw):
        try:
            result.append(ExtractedMemory.model_validate(item))
        except ValidationError:
            log.warning("extraction_dropped_invalid_candidate", candidate=item)
    return result
