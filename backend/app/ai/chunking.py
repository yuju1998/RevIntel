"""V1 chunking: one chunk per source unless the text exceeds the model's token limit.

Current CRM texts are short (tens of characters), so each stays a single, unchanged chunk.
Longer text is split at whitespace into windows of at most `max_tokens` tokens, measured
with the embedding model's own tokenizer, so the model never silently truncates a chunk.
Consecutive windows share about `overlap_tokens` tokens of context. Inner whitespace of
each window is preserved (windows are slices of the original text).
"""

import re
from collections.abc import Callable

DEFAULT_OVERLAP_TOKENS = 32
_WORD = re.compile(r"\S+")


def chunk_text(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    # Words as (start, end) spans; a single word longer than the limit is split by characters.
    spans: list[tuple[int, int]] = []
    for match in _WORD.finditer(text):
        spans.extend(_split_long_word(text, match.start(), match.end(), count_tokens, max_tokens))
    sizes = [count_tokens(text[start:end]) for start, end in spans]

    chunks: list[str] = []
    first = 0
    while first < len(spans):
        # Greedily extend the window by word token counts, then confirm with the tokenizer
        # on the actual slice (shrinking if word-by-word counting underestimated it).
        last, total = first, sizes[first]
        while last + 1 < len(spans) and total + sizes[last + 1] <= max_tokens:
            last += 1
            total += sizes[last]
        while last > first and count_tokens(text[spans[first][0] : spans[last][1]]) > max_tokens:
            last -= 1
        chunks.append(text[spans[first][0] : spans[last][1]])
        if last == len(spans) - 1:
            break
        # Next window starts early enough to repeat ~overlap_tokens, but always advances.
        start, carried = last, 0
        while start > first and carried + sizes[start] <= overlap_tokens:
            carried += sizes[start]
            start -= 1
        first = max(start + 1, first + 1) if carried else last + 1
    return chunks


def _split_long_word(
    text: str, start: int, end: int, count_tokens: Callable[[str], int], max_tokens: int
) -> list[tuple[int, int]]:
    if count_tokens(text[start:end]) <= max_tokens or end - start <= 1:
        return [(start, end)]
    middle = (start + end) // 2
    return _split_long_word(text, start, middle, count_tokens, max_tokens) + _split_long_word(
        text, middle, end, count_tokens, max_tokens
    )
