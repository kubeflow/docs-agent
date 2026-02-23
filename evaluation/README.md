# Retrieval Evaluation Framework

This directory contains a suite of tools to evaluate the performance of the Kubeflow Documentation AI Assistant.

## Components

1.  **`config.py`**: Central configuration for Milvus, embedding models, and LLM endpoints.
2.  **`generate_dataset.py`**: Generates a synthetic evaluation dataset (Question, Answer, Context) from the existing Milvus index using an LLM.
3.  **`eval_retrieval.py`**: Evaluates the retrieval system (Milvus search) using metrics like Hit Rate and Mean Reciprocal Rank (MRR).
4.  **`eval_rag.py`**: Evaluates the end-to-end RAG pipeline using the [Ragas](https://github.com/explodinggradients/ragas) framework.

## Setup

1.  Install the required dependencies:
    ```bash
    pip install -r evaluation/requirements.txt
    ```
2.  Configure your environment variables in `.env` or update `evaluation/config.py`. Ensure you have access to:
    -   A running Milvus instance.
    -   An OpenAI-compatible LLM endpoint (e.g., KServe, vLLM, or OpenAI).

## Usage

### 1. Generate Evaluation Dataset
First, generate a synthetic dataset from your actual documentation content:
```bash
python evaluation/generate_dataset.py
```
This will create `evaluation/dataset.jsonl`.

### 2. Evaluate Retrieval
Run the retrieval evaluation to see how well the system finds relevant documentation:
```bash
python evaluation/eval_retrieval.py
```

### 3. Evaluate End-to-End RAG
Run the RAG evaluation to measure the quality of the generated answers (Faithfulness, Relevancy, etc.):
```bash
python evaluation/eval_rag.py
```

## Metrics Explanation

-   **Hit Rate**: The percentage of queries where the "correct" document was found in the top-k results.
-   **MRR (Mean Reciprocal Rank)**: Measures the rank of the first relevant document.
-   **Faithfulness**: Measures if the answer is derived solely from the provided context.
-   **Answer Relevancy**: Measures how relevant the answer is to the question.
-   **Context Precision**: Measures how well the relevant documents are ranked in the context.
-   **Context Recall**: Measures if all relevant information is present in the context.
