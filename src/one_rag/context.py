from one_rag.schemas import Evidence


def build_context(evidence: list[Evidence] | list[dict[str, object]]) -> str:
    """Keep expanded parent context once, while retaining every child as evidence."""
    blocks = []
    included = set()
    for item in evidence:
        value = item.model_dump() if isinstance(item, Evidence) else item
        key = (value["document_id"], value.get("parent_chunk_index")) if value.get("parent_text") else (value["document_id"], value["chunk_index"])
        if key in included:
            continue
        included.add(key)
        label = f"parent {value['parent_chunk_index']}" if value.get("parent_text") else f"chunk {value['chunk_index']}"
        blocks.append(f"[{value['source']} | {label}]\n{value.get('parent_text') or value['text']}")
    return "\n\n".join(blocks)
