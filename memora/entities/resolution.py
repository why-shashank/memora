"""Canonicalizing a surface form into the key the alias index is looked up by.

Alias matching only — no embedding-based scoring. S5 measured the assumption behind
that choice: surface-form consistency was the extraction model's *strongest* result
(`Priya Nair` -> `Priya` -> `P. Nair`, and `T. Okafor` -> `Tomas Okafor`, both 3/3
reps), so the fuzzy matching this was hedging against has nothing to fix. Embeddings
would also make the cross-type collapse in `store.resolve_entity` worse rather than
better: a person and their employer co-occur in every sentence, so they embed close
together.
"""

import re

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")

# Stripped so 'Lumen Health Inc.' and 'Lumen Health' resolve to one customer. S5
# measured this as the difference between 11/12 and 12/12 cross-conversation merges.
# Matched as whole words only, so 'BrightCo' and 'Cisco' keep their endings.
_LEGAL_SUFFIX = re.compile(
    r"\b(inc|incorporated|ltd|limited|llc|llp|corp|corporation|co|plc|gmbh)\b"
)


def canonical_key(surface_form: str) -> str:
    """Fold case, punctuation, whitespace and legal suffixes into a lookup key.

    Punctuation becomes a space rather than nothing, so 'jane.doe@acme.co' keys on
    its parts instead of fusing into one run of letters.
    """
    key = _PUNCTUATION.sub(" ", surface_form.lower())
    stripped = _WHITESPACE.sub(" ", _LEGAL_SUFFIX.sub("", key)).strip()
    # A name that is *only* a suffix ('Co') would strip to empty and then collide
    # with every other empty key, so it keeps its unstripped form.
    return stripped or _WHITESPACE.sub(" ", key).strip()


# Longest name the entity leg will look for. Four words covers 'Bank of the West' and
# 'Acme Freight International'; going wider costs a key per extra word per position and
# finds names nobody types into a question.
_MAX_NAME_WORDS = 4


def candidate_keys(query_text: str, *, max_words: int = _MAX_NAME_WORDS) -> set[str]:
    """Every short run of words in ``query_text``, keyed the way an alias is keyed.

    Retrieval has no extraction step, so it cannot know which words of a question are a
    name — offering all of them and letting the alias index reject the rest is cheaper
    than an LLM call on the read path, and it is exact: a run of words either is a name
    somebody stored or it is not.

    The whole question is canonicalized *first* and then split, so the folding both sides
    depend on happens once and identically. The precision risk is real and worth watching:
    a person called 'Will' or a product called 'Chat' turns a common word into an entity
    match, and every question containing it fires the leg.
    """
    words = canonical_key(query_text).split()
    return {
        " ".join(words[start : start + length])
        for length in range(1, max_words + 1)
        for start in range(len(words) - length + 1)
    }
