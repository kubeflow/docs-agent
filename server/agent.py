import os
import json
import logging
import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Dict, Any

from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolNode, tools_condition

# Import existing search logic
from tools import milvus_search, search_github_issues

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# KServe OpenAI-compatible endpoint
# The client expects base_url to be the root of the API, e.g. .../v1
# If KSERVE_URL is .../v1/chat/completions, we strip the suffix
KSERVE_URL = os.getenv("KSERVE_URL", "http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions")
BASE_URL = KSERVE_URL.replace("/chat/completions", "")
MODEL_NAME = os.getenv("MODEL", "llama3.1-8B")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY") # Local/KServe often needs a dummy key

# --- Tools ---

@tool
def search_kubeflow_docs(query: str) -> str:
    """
    Search the official Kubeflow documentation.
    Use this tool when the user asks specific questions about Kubeflow components,
    installation, configuration, or usage.
    """
    try:
        # Default top_k to 5
        results = milvus_search(query, top_k=5)
        hits = results.get("results", [])
        if not hits:
            return "No relevant documentation found."
        
        formatted_results = []
        for hit in hits:
            formatted_results.append(
                f"Source: {hit.get('citation_url', 'Unknown')}\n"
                f"Content: {hit.get('content_text', '')}\n"
            )
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching docs: {str(e)}"

tools = [search_kubeflow_docs, search_github_issues]
# ToolNode is a prebuilt node in LangGraph for executing tools
tool_node = ToolNode(tools)

# --- Model ---

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
    streaming=True
)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

# --- State ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- Nodes ---

def agent_node(state: AgentState):
    """
    Invokes the LLM to decide the next step.
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- Graph Definition ---

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# tools_condition Checks for tool_calls in the last message
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

workflow.add_edge("tools", "agent")

# Compile the graph
agent_graph = workflow.compile()
