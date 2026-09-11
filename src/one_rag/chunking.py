import re


SUPPORTED_STRATEGIES = {"fixed", "fixed_overlap", "sentence", "sentence_window", "paragraph", "section", "parent_child"}


def token_count(text: str) -> int:
    """Count lexical tokens so chunk limits are deterministic and inspectable."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def sentence_chunks(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def fixed_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return [" ".join(tokens[start : start + size]) for start in range(0, len(tokens), size - overlap)]


def paragraph_chunks(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def section_chunks(text: str) -> list[str]:
    sections: list[str] = []
    heading = ""
    body: list[str] = []
    for line in text.strip().splitlines():
        if re.match(r"^(#{1,6}\s+|\d{1,2}\.\s+[A-Z][A-Z ]+$)", line):
            if heading or body:
                sections.append("\n".join([heading, *body]).strip())
            heading, body = line.strip(), []
        else:
            body.append(line)
    if heading or body:
        sections.append("\n".join([heading, *body]).strip())
    return [section for section in sections if section]


def sentence_window_chunks(text: str, size: int, overlap: int) -> list[str]:
    sentences = sentence_chunks(text)
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Sentence window size must be positive and overlap must be smaller than it.")
    return [" ".join(sentences[start : start + size]) for start in range(0, len(sentences), size - overlap)]


def chunk_text(text: str, strategy: str = "sentence", chunk_size: int = 500, chunk_overlap: int = 100, sentence_window_size: int = 2, sentence_window_overlap: int = 1) -> list[str]:
    """Split text using an explicit, inspectable Stage 2 strategy."""
    if strategy not in SUPPORTED_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(f"Unsupported chunking strategy '{strategy}'. Choose one of: {choices}.")
    if strategy == "fixed":
        return fixed_chunks(text, chunk_size)
    if strategy == "fixed_overlap":
        return fixed_chunks(text, chunk_size, chunk_overlap)
    if strategy == "sentence":
        return sentence_chunks(text)
    if strategy == "sentence_window":
        return sentence_window_chunks(text, sentence_window_size, sentence_window_overlap)
    if strategy == "paragraph":
        return paragraph_chunks(text)
    if strategy == "section":
        return section_chunks(text)
    raise ValueError("Parent-child chunking is handled by the retrieval service.")
