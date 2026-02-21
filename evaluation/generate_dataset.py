import json
import random
from pymilvus import connections, Collection
from openai import OpenAI
from tqdm import tqdm
from config import *

def get_random_samples(n=10):
    """Fetch random samples from Milvus to generate evaluation data."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    
    # Get total number of entities
    total = collection.num_entities
    if total == 0:
        print("Collection is empty!")
        return []
    
    # Milvus doesn't have a direct 'sample' but we can use offset if IDs are sequential or just fetch a block
    # For simplicity, we'll fetch the first 100 and sample from them, or use a large limit
    results = collection.query(
        expr="", 
        output_fields=["content_text", "citation_url", "file_path"],
        limit=min(total, 100)
    )
    
    connections.disconnect(alias="default")
    return random.sample(results, min(len(results), n))

def generate_triplet(client, context):
    """Use LLM to generate a Question and Answer from a given context."""
    prompt = f"""
    Given the following documentation context, generate a specific question that can be answered using ONLY this context.
    Also provide the answer based on the context.
    
    Context: {context}
    
    Return the result in JSON format:
    {{
        "question": "...",
        "answer": "..."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating triplet: {e}")
        return None

def main():
    client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
    
    print(f"Fetching samples from Milvus ({MILVUS_COLLECTION})...")
    samples = get_random_samples(20)
    
    dataset = []
    print("Generating synthetic questions and answers...")
    for sample in tqdm(samples):
        context = sample['content_text']
        triplet = generate_triplet(client, context)
        if triplet:
            dataset.append({
                "question": triplet['question'],
                "ground_truth": triplet['answer'],
                "context": context,
                "citation_url": sample['citation_url'],
                "file_path": sample['file_path']
            })
            
    with open(DATASET_PATH, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"Dataset saved to {DATASET_PATH} ({len(dataset)} items)")

if __name__ == "__main__":
    main()
