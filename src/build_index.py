"""Step 3: turn chunks into embeddings and store them in ChromaDB.

Usage:
    python src/build_index.py --chunk-size 1000
"""
import argparse
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
DB_DIR = "chroma_db"
BATCH = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    in_file = Path(f"data/processed/chunks_{args.chunk_size}.jsonl")
    chunks = [json.loads(line) for line in in_file.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks from {in_file}")

    model = SentenceTransformer(MODEL_NAME)
    start = time.time()
    # e5 needs "passage: " in front of every document text
    vectors = model.encode(
        ["passage: " + c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,   # length 1, so distance ranking = cosine ranking
        show_progress_bar=True,
    )
    embed_seconds = time.time() - start

    client = chromadb.PersistentClient(path=DB_DIR)
    name = f"chunks_{args.chunk_size}"
    try:
        client.delete_collection(name)   # rebuild from scratch every time
    except Exception:
        pass
    collection = client.create_collection(name)

    for i in range(0, len(chunks), BATCH):
        part = chunks[i:i + BATCH]
        collection.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=[v.tolist() for v in vectors[i:i + BATCH]],
            metadatas=[{"doc": c["doc"], "lang": c["lang"], "page": c["page"]} for c in part],
        )

    print(f"Embedding time: {embed_seconds:.1f} s "
          f"({embed_seconds / len(chunks) * 1000:.1f} ms per chunk)")
    print(f"Stored {collection.count()} chunks in collection '{name}' ({DB_DIR}/)")


if __name__ == "__main__":
    main()