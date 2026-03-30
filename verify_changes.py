content = open('pipelines/kubeflow-pipeline.py', encoding='utf-8').read()

if 'h1": record.get("h1' in content:
    print('A3 OK - h1/h2/h3 fields in records.append')
else:
    print('A3 MISSING - h1/h2/h3 not found in records.append')

if 'docs_index' in content:
    print('A4 OK - collection renamed to docs_index')
else:
    print('A4 MISSING - still says docs_rag')

if 'MarkdownHeaderTextSplitter' in content:
    print('A1 OK - MarkdownHeaderTextSplitter present')
else:
    print('A1 MISSING - chunker not upgraded')