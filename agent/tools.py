

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kubeflow_docs",
            "description": (
                "Search the official Kubeflow docs when the user asks Kubeflow-specific questions "
                "about Pipelines, KServe, Notebooks/Jupyter, Katib, or the SDK/CLI/APIs.\n"
                "Call ONLY for Kubeflow features, setup, usage, errors, or version differences that need citations.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short, focused search string (e.g., 'KServe inferenceService canary', 'Pipelines v2 disable cache').",
                        "minLength": 1
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of hits to retrieve (the assistant will read up to this many).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]
