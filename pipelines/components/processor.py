from kfp import dsl
from kfp.dsl import Input, Output, Dataset

@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers", "langchain"]
)
def document_processor_component(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
):
    import json
    import os
    import re
    import torch
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(embedding_model, device=device)
    print(f"Model loaded on {device}")

    records = []

    with open(input_dataset.path, 'r', encoding='utf-8') as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data['content']
            path = file_data['path']
            repo = file_data.get('repo', 'kubeflow/website')

            # --- Aggressive Cleaning ---
            content = re.sub(r'^\s*[+\-]{3,}.*?[+\-]{3,}\s*', '', content, flags=re.DOTALL | re.MULTILINE)
            content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)
            content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()

            if len(content) < 50:
                continue

            # --- Citation URL Mapping ---
            # Custom logic for Kubeflow docs website structure
            if 'content/en/docs' in path:
                path_parts = path.split('/')
                docs_index = path_parts.index('docs')
                url_path = '/'.join(path_parts[docs_index+1:])
                url_path = os.path.splitext(url_path)[0]
                citation_url = f"{base_url}/{url_path}"
            else:
                citation_url = f"{base_url}/{path}"

            # --- Chunking ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(content)

            # --- Embedding ---
            for chunk_idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                records.append({
                    'file_unique_id': f"{repo}:{path}",
                    'repo_name': repo,
                    'file_path': path,
                    'citation_url': citation_url,
                    'chunk_index': chunk_idx,
                    'content_text': chunk,
                    'embedding': embedding
                })

    with open(output_dataset.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
