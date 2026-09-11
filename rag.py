import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "sentence")
SENTENCE_WINDOW_SIZE = int(os.getenv("SENTENCE_WINDOW_SIZE", "2"))
SENTENCE_WINDOW_OVERLAP = int(os.getenv("SENTENCE_WINDOW_OVERLAP", "1"))


def sentence_chunks(text):
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE).strip()
    return [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]


def sentence_window_chunks(text):
    sentences = sentence_chunks(text)
    step = SENTENCE_WINDOW_SIZE - SENTENCE_WINDOW_OVERLAP
    if step < 1:
        raise ValueError("SENTENCE_WINDOW_OVERLAP must be smaller than SENTENCE_WINDOW_SIZE.")
    windows = []
    for start in range(0, len(sentences), step):
        window = sentences[start:start + SENTENCE_WINDOW_SIZE]
        if len(window) < SENTENCE_WINDOW_SIZE and windows:
            break
        windows.append(" ".join(window))
    return windows


def section_chunks(text):
    sections = []
    chunks = []
    for line in text.splitlines():
        if line.startswith("#") and chunks:
            sections.append("\n".join(chunks).strip())
            chunks = []
        chunks.append(line)
    if chunks:
        sections.append("\n".join(chunks).strip())
    return sections


def paragraph_chunks(text):
    chunks = []
    heading = ""
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("#"):
            heading = chunk
        elif chunk:
            chunks.append(f"{heading}\n\n{chunk}".strip())
            heading = ""
    return chunks


def chunk_text(text):
    strategies = {
        "sentence": sentence_chunks,
        "sentence_window": sentence_window_chunks,
        "section": section_chunks,
        "paragraph": paragraph_chunks,
    }
    try:
        return strategies[CHUNKING_STRATEGY](text)
    except KeyError:
        raise ValueError("CHUNKING_STRATEGY must be sentence, sentence_window, section, or paragraph.")


def load_documents(folder):
    documents = []
    for path in sorted(Path(folder).glob("*.txt")):
        for index, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            documents.append({"source": path.name, "chunk": index, "text": chunk})
    return documents


def index(folder):
    documents = load_documents(folder)
    if not documents:
        raise ValueError("Add at least one .txt file to the documents folder.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([document["text"] for document in documents]).tolist()
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=index, vector=vector, payload=document) for index, (vector, document) in enumerate(zip(vectors, documents))],
    )
    print(f"Indexed {len(documents)} chunks.")


def ask(question, limit):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    results = client.query_points(
        collection_name=COLLECTION,
        query=model.encode(question).tolist(),
        limit=limit,
    ).points
    for result in results:
        print(f"\n[{result.payload['source']} | chunk {result.payload['chunk']} | score {result.score:.3f}]")
        print(result.payload["text"])


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index")
    index_parser.add_argument("folder", nargs="?", default="documents")
    ask_parser = commands.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.command == "index":
        index(args.folder)
    else:
        ask(args.question, args.limit)


if __name__ == "__main__":
    main()
