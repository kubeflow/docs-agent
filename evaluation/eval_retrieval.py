import json
import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import *

def milvus_search(query, top_k=5):
    """Simple search function for evaluation."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = encoder.encode(query).tolist()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": MILVUS_NPROBE}}
    results = collection.search(
        data=[query_vec],
        anns_field=MILVUS_VECTOR_FIELD,
        param=search_params,
        limit=top_k,
        output_fields=["content_text", "file_path"]
    )
    
    hits = []
    for hit in results[0]:
        hits.append({
            "content": hit.entity.get("content_text"),
            "file_path": hit.entity.get("file_path")
        })
    
    connections.disconnect(alias="default")
    return hits

def evaluate_retrieval():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first.")
        return

    dataset = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    hit_count = 0
    rr_sum = 0
    
    print(f"Evaluating retrieval for {len(dataset)} queries...")
    for item in tqdm(dataset):
        query = item['question']
        expected_context = item['context']
        
        results = milvus_search(query, top_k=TOP_K)
        
        # Check if expected context (or same file) is in results
        found = False
        for rank, res in enumerate(results):
            # We use a simple strategy: if the expected context is a substring or identical
            if expected_context in res['content'] or res['content'] in expected_context:
                hit_count += 1
                rr_sum += 1.0 / (rank + 1)
                found = True
                break
    
    hit_rate = hit_count / len(dataset) if dataset else 0
    mrr = rr_sum / len(dataset) if dataset else 0
    
    print("\n--- Retrieval Evaluation Results ---")
    print(f"Total Queries: {len(dataset)}")
    print(f"Hit Rate @{TOP_K}: {hit_rate:.4f}")
    print(f"MRR @{TOP_K}:      {mrr:.4f}")
    
    return {"hit_rate": hit_rate, "mrr": mrr}

if __name__ == "__main__":
    evaluate_retrieval()
