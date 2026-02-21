import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from openai import OpenAI
from eval_retrieval import milvus_search
from config import *

# Wrap the internal search and LLM to work with Ragas or just use our own evaluation loop
# Ragas expects a dataset with: question, contexts, answer, ground_truth

def get_assistant_response(client, question):
    """Fetch response from the docs assistant server."""
    # We can either call the local server or use the same logic as the server
    # For evaluation, it's often better to call the actual endpoint if possible
    # but for simplicity, we'll mimic the internal RAG call here
    
    results = milvus_search(question, top_k=TOP_K)
    context = "\n\n".join([r['content'] for r in results])
    
    prompt = f"Given the context: {context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response.choices[0].message.content,
            "contexts": [r['content'] for r in results]
        }
    except Exception as e:
        print(f"Error getting assistant response: {e}")
        return {"answer": "Error", "contexts": []}

def evaluate_rag():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first.")
        return

    client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
    
    dataset_raw = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            dataset_raw.append(json.loads(line))

    # Prepare data for Ragas
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Generating assistant responses for {len(dataset_raw)} queries...")
    for item in dataset_raw:
        resp = get_assistant_response(client, item['question'])
        data["question"].append(item['question'])
        data["answer"].append(resp['answer'])
        data["contexts"].append(resp['contexts'])
        data["ground_truth"].append(item['ground_truth'])

    hf_dataset = Dataset.from_dict(data)
    
    print("Running Ragas evaluation...")
    # metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    # Note: Ragas evaluation might require an LLM for scoring (using OpenAIGPT4 or similar)
    # If the user has a local LLM serving OpenAI-compatible API, we can use it.
    
    result = evaluate(
        hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    print("\n--- RAG Evaluation Results ---")
    df = result.to_pandas()
    print(df.mean(numeric_only=True))
    
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Detailed results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_rag()
