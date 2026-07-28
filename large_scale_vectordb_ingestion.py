import os
import asyncio
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Connect to ChromaDB as an independent HTTP Server (Production setup)
# (e.g., Chroma running in a Docker container or remote server)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="production_knowledge_base")

# 2. Load model on GPU if available
device = "cuda" if os.environ.get("USE_GPU") else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Simple text chunker with overlap
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_insert_batch(docs: List[str], ids: List[str]):
    """Embeds and inserts a single batch safely."""
    # Batch GPU/CPU inference
    embeddings = embedding_model.encode(docs, batch_size=64, show_progress_bar=False).tolist()
    
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids
    )

def production_ingest(file_paths: List[str], batch_size: int = 1000):
    batch_docs = []
    batch_ids = []
    global_counter = 0

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # STEP A: Chunk raw documents
        doc_chunks = chunk_text(raw_text)

        for chunk in doc_chunks:
            batch_docs.append(chunk)
            batch_ids.append(f"doc_{global_counter}")
            global_counter += 1

            # STEP B: Flush batch when limit reached
            if len(batch_docs) >= batch_size:
                process_and_insert_batch(batch_docs, batch_ids)
                print(f"[Ingestion]: Processed {global_counter} chunks...")
                batch_docs.clear()
                batch_ids.clear()

    # Flush remaining
    if batch_docs:
        process_and_insert_batch(batch_docs, batch_ids)
        print(f"[Ingestion Complete]: Total {global_counter} chunks indexed.")

if __name__ == "__main__":
    # Run out-of-process from your main web application
    files_to_index = ["amazon_policy_1.txt", "amazon_policy_2.txt"]
    production_ingest(files_to_index)