import logging
import random
import time
from typing import TypedDict, Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

# --- Constants ---
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CoreAgent")

# --- State Schema ---
class AgentState(TypedDict):
    user_query: str
    intent: str
    context: str
    retrieved_context: List[str]
    citations: List[str]
    response: str
    generation: str
    error_count: int

# --- Helper functions for "external" calls (to be mocked in tests) ---

def call_llm_classify(query: str) -> str:
    """Mockable LLM call for classification."""
    if any(word in query.lower() for word in ["error", "log", "debug", "fail"]):
        return 'Debugging / Error Log'
    elif any(word in query.lower() for word in ["pipeline", "config", "yaml", "setup"]):
        return 'Pipeline Configuration'
    elif any(word in query.lower() for word in ["what", "how", "define", "concept"]):
        return 'Conceptual Definition'
    elif any(word in query.lower() for word in ["api", "endpoint", "reference", "parameter"]):
        return 'API Reference'
    return 'General Conversation'

def call_milvus_search(query: str, partition: str) -> Dict[str, Any]:
    """Mockable Milvus search call."""
    return {
        "content": f"Retrieved {partition} content for '{query}'.",
        "citations": [f"https://docs.kubeflow.org/{partition}/test"]
    }

def call_llm_generate(intent: str, query: str, context: str) -> str:
    """Mockable LLM call for generation."""
    if random.random() < 0.2: 
        return ""
    return f"This is a generated response for the query: '{query}' based on {intent}."

# --- Nodes ---

def classify_intent(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    intent = call_llm_classify(state["user_query"])
    latency = (time.time() - start_time) * 1000
    logger.info(f"[Node: classify_intent] Intent: {intent} | Latency: {latency:.2f}ms")
    return {"intent": intent}

def retrieve_docs(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"[Node: retrieve_docs] Partition: Documentation | Query: {state['user_query']}")
    res = call_milvus_search(state["user_query"], "docs")
    latency = (time.time() - start_time) * 1000
    logger.info(f"[Node: retrieve_docs] Completed | Latency: {latency:.2f}ms")
    return {
        "context": res["content"],
        "retrieved_context": [res["content"]],
        "citations": res["citations"]
    }

def retrieve_github_issues(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"[Node: retrieve_github_issues] Partition: GitHub Issues | Query: {state['user_query']}")
    res = call_milvus_search(state["user_query"], "github")
    latency = (time.time() - start_time) * 1000
    logger.info(f"[Node: retrieve_github_issues] Completed | Latency: {latency:.2f}ms")
    return {
        "context": res["content"],
        "retrieved_context": [res["content"]],
        "citations": res["citations"]
    }

def retrieve_architecture(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"[Node: retrieve_architecture] Partition: Architecture/Pipelines | Query: {state['user_query']}")
    res = call_milvus_search(state["user_query"], "architecture")
    latency = (time.time() - start_time) * 1000
    logger.info(f"[Node: retrieve_architecture] Completed | Latency: {latency:.2f}ms")
    return {
        "context": res["content"],
        "retrieved_context": [res["content"]],
        "citations": res["citations"]
    }

def generate_response(state: AgentState) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"[Node: generate_response] Intent: {state['intent']} | Error Count: {state['error_count']}")
    
    gen = call_llm_generate(state["intent"], state["user_query"], state["context"])
    error_inc = 1 if not gen else 0
    
    if not gen:
        logger.warning("[Node: generate_response] Failure: Empty response generated.")
    else:
        logger.info("[Node: generate_response] Success: Response generated.")
        
    latency = (time.time() - start_time) * 1000
    logger.info(f"[Node: generate_response] Completed | Latency: {latency:.2f}ms")
    
    return {
        "response": gen,
        "generation": gen,
        "error_count": state["error_count"] + error_inc
    }

# --- Routing Logic ---

def route_after_classification(state: AgentState) -> str:
    intent = state["intent"]
    if intent == 'Debugging / Error Log':
        return "retrieve_github_issues"
    elif intent == 'Pipeline Configuration':
        return "retrieve_architecture"
    elif intent in ['Conceptual Definition', 'API Reference']:
        return "retrieve_docs"
    else:
        return "generate_response"

def route_after_generation(state: AgentState) -> str:
    # If the response is empty (failure) and error_count < MAX_RETRIES, retry
    if not state.get("generation") and state.get("error_count", 0) < MAX_RETRIES:
        logger.info(f"[Routing] Response flagged as failure. Error count: {state['error_count']}. Retrying...")
        return route_after_classification(state) # Directly return the next node
    
    if not state.get("generation"):
         logger.error(f"[Routing] Max retries reached ({MAX_RETRIES}). Bailing out.")
    return END

# --- Graph Construction ---

def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("retrieve_github_issues", retrieve_github_issues)
    workflow.add_node("retrieve_architecture", retrieve_architecture)
    workflow.add_node("generate_response", generate_response)

    # Set Entry Point
    workflow.set_entry_point("classify_intent")

    # Conditional Edges after classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_github_issues": "retrieve_github_issues",
            "retrieve_architecture": "retrieve_architecture",
            "retrieve_docs": "retrieve_docs",
            "generate_response": "generate_response"
        }
    )

    # Edges from retrieval nodes to generation
    workflow.add_edge("retrieve_docs", "generate_response")
    workflow.add_edge("retrieve_github_issues", "generate_response")
    workflow.add_edge("retrieve_architecture", "generate_response")

    # Conditional Edges after generate_response (Cyclic Error Correction)
    workflow.add_conditional_edges(
        "generate_response",
        route_after_generation,
        {
            "retrieve_docs": "retrieve_docs",
            "retrieve_github_issues": "retrieve_github_issues",
            "retrieve_architecture": "retrieve_architecture",
            "generate_response": "generate_response",
            END: END
        }
    )

    return workflow.compile()

agent_graph = create_graph()
