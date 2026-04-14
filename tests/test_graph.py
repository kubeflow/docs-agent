import pytest
from unittest.mock import patch, MagicMock
from core_agent.graph import agent_graph, MAX_RETRIES

# --- Fixtures ---

@pytest.fixture
def initial_state():
    """Returns a clean initial state for the agent."""
    return {
        "user_query": "Test query",
        "intent": "",
        "context": "",
        "retrieved_context": [],
        "citations": [],
        "response": "",
        "generation": "",
        "error_count": 0
    }

# --- Parameterized Routing Tests ---

@pytest.mark.parametrize("mock_intent, expected_retrieval_node", [
    ('Conceptual Definition', 'retrieve_docs'),
    ('API Reference', 'retrieve_docs'),
    ('Debugging / Error Log', 'retrieve_github_issues'),
    ('Pipeline Configuration', 'retrieve_architecture'),
])
def test_routing_to_retrieval(initial_state, mock_intent, expected_retrieval_node):
    """
    Tests that the graph correctly routes from classify_intent to the 
    appropriate retrieval node based on the intent.
    """
    with patch('core_agent.graph.call_llm_classify', return_value=mock_intent), \
         patch('core_agent.graph.call_milvus_search') as mock_milvus, \
         patch('core_agent.graph.call_llm_generate') as mock_gen:
        
        # Mock Milvus to return dummy data
        mock_milvus.return_value = {
            "content": "Dummy content",
            "citations": ["http://test.com"]
        }
        # Mock LLM generation to return a successful response
        mock_gen.return_value = "Successful response"
        
        # Execute the graph
        final_state = agent_graph.invoke(initial_state)
        
        # Assertions
        assert final_state['intent'] == mock_intent
        assert final_state['generation'] == "Successful response"
        # Verify the correct retrieval node was called (via its side effect on the state)
        if expected_retrieval_node == 'retrieve_docs':
            assert "docs" in str(mock_milvus.call_args_list)
        elif expected_retrieval_node == 'retrieve_github_issues':
            assert "github" in str(mock_milvus.call_args_list)
        elif expected_retrieval_node == 'retrieve_architecture':
            assert "architecture" in str(mock_milvus.call_args_list)

def test_routing_general_conversation(initial_state):
    """
    Tests that 'General Conversation' intent bypasses retrieval nodes 
    and goes directly to generate_response.
    """
    mock_intent = 'General Conversation'
    with patch('core_agent.graph.call_llm_classify', return_value=mock_intent), \
         patch('core_agent.graph.call_milvus_search') as mock_milvus, \
         patch('core_agent.graph.call_llm_generate', return_value="General response"):
        
        final_state = agent_graph.invoke(initial_state)
        
        assert final_state['intent'] == mock_intent
        assert final_state['generation'] == "General response"
        # Milvus search should NOT be called for general conversation
        mock_milvus.assert_not_called()

# --- Cyclic Error Correction Tests ---

def test_cyclic_error_correction_loop(initial_state):
    """
    Tests the failure recovery loop. 
    Simulates a single failure in generation, asserts retry, and then a success.
    """
    mock_intent = 'API Reference'
    
    # We want call_llm_generate to return "" first (failure), then "Success"
    side_effects = ["", "Success after retry"]
    
    with patch('core_agent.graph.call_llm_classify', return_value=mock_intent), \
         patch('core_agent.graph.call_milvus_search', return_value={"content": "data", "citations": []}), \
         patch('core_agent.graph.call_llm_generate', side_effect=side_effects) as mock_gen:
        
        final_state = agent_graph.invoke(initial_state)
        
        # Verify it called generation twice
        assert mock_gen.call_count == 2
        # Verify error_count was incremented then carried over
        assert final_state['error_count'] == 1 # 1 failure, then 1 success (which doesn't inc)
        assert final_state['generation'] == "Success after retry"
        assert final_state['intent'] == mock_intent

# --- Bailout Condition Test ---

def test_bailout_condition(initial_state):
    """
    Tests that the graph exits to END when error_count reaches MAX_RETRIES.
    Ensures no infinite loops.
    """
    mock_intent = 'Conceptual Definition'
    
    # Always return failure
    with patch('core_agent.graph.call_llm_classify', return_value=mock_intent), \
         patch('core_agent.graph.call_milvus_search', return_value={"content": "data", "citations": []}), \
         patch('core_agent.graph.call_llm_generate', return_value="") as mock_gen:
        
        final_state = agent_graph.invoke(initial_state)
        
        # Should have called generate MAX_RETRIES times
        assert mock_gen.call_count == MAX_RETRIES
        assert final_state['error_count'] == MAX_RETRIES
        assert final_state['generation'] == "" # Or a fallback message if implemented
        # Verify we didn't get stuck in an infinite loop
        # (The fact that invoke() returned is proof enough in a synchronous test)

# --- Robustness / Edge Case Tests ---

def test_initial_error_count_high(initial_state):
    """
    Tests behavior when starting with an already high error count.
    """
    initial_state['error_count'] = MAX_RETRIES
    with patch('core_agent.graph.call_llm_classify', return_value='General Conversation'), \
         patch('core_agent.graph.call_llm_generate', return_value=""):
        
        final_state = agent_graph.invoke(initial_state)
        
        # Even if it fails, it should exit immediately because error_count is already at max
        assert final_state['error_count'] == MAX_RETRIES + 1
        assert final_state['generation'] == ""
