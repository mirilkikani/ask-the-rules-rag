"""Quick test: find the 5 most similar chunks for a question."""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
question = " ".join(sys.argv[1:]) or "What is AI literacy?"

model = SentenceTransformer(MODEL_NAME)
vec = model.encode("query: " + question, normalize_embeddings=True)  # e5 needs "query: "

collection = chromadb.PersistentClient(path="chroma_db").get_collection("chunks_1000")
res = collection.query(query_embeddings=[vec.tolist()], n_results=5)

print(f"Question: {question}\n")
for rank, (text, meta, dist) in enumerate(
    zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
):
    print(f"{rank}. {meta['doc']} p.{meta['page']} (distance {dist:.3f})")
    print(f"   {text[:150]}...\n")