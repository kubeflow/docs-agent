import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies before importing agent
sys.modules['pymilvus'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()

# Mock ChatOpenAI specifically
mock_llm = MagicMock()
sys.modules['langchain_openai'].ChatOpenAI.return_value = mock_llm

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph

# Import agent (this will trigger graph compilation)
# We need to patch tools because they are imported in agent.py
with patch('server.tools.milvus_search') as mock_milvus, \
     patch('server.tools.search_github_issues') as mock_github:
    
    from server.agent import agent_graph, tools

class TestAgentGraph(unittest.TestCase):
    
    def test_graph_compilation(self):
        """Test that the graph compiles successfully"""
        self.assertIsNotNone(agent_graph)
        
    def test_tools_registered(self):
        """Test that tools are correct"""
        tool_names = [t.name for t in tools]
        self.assertIn('search_kubeflow_docs', tool_names)
        self.assertIn('search_github_issues', tool_names)

    @patch('server.agent.llm_with_tools')
    def test_agent_decision(self, mock_llm_with_tools):
        """Test agent node execution"""
        # Prepare state
        state = {"messages": [HumanMessage(content="help me with kubeflow")]}
        
        # Mock LLM response
        mock_response = AIMessage(content="Sure!")
        mock_llm_with_tools.invoke.return_value = mock_response
        
        # Run agent node (we can't easily run the graph without full mocks, but we can test components)
        from server.agent import agent_node
        result = agent_node(state)
        
        self.assertEqual(result['messages'][0].content, "Sure!")

if __name__ == '__main__':
    unittest.main()
