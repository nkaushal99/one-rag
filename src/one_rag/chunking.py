import re


def sentence_chunks(text: str) -> list[str]:
    cleaned = re.sub(r"^#.*$", "", text, flags=re.MULTILINE).strip()
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", cleaned) if chunk.strip()]


def chunk_text(text: str) -> list[str]:
    """Keep Stage 1 chunks small and sentence-aligned for easy inspection."""
    return sentence_chunks(text)
