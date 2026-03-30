from agent.core.graph import agent

questions = [
    "What is Kubeflow Pipelines and how does it work?",
    "How do I fix a CrashLoopBackOff error in my KServe deployment?",
    "Show me the webhook configuration YAML for notebooks",
]

for q in questions:
    print(f"\nQ: {q}")
    result = agent.invoke({
        "question":  q,
        "route":     "",
        "chunks":    [],
        "citations": [],
        "answer":    "",
        "messages":  []
    })
    print(f"Route:  {result['route']}")
    print(f"Chunks: {len(result['chunks'])}")
    print(f"Answer: {result['answer'][:200]}")
    print("-" * 50)