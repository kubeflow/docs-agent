from agent.tools.mcp_tools import query_docs_tool, query_code_tool, query_both_tool
import json

print("=" * 50)
print("TEST 1 - query_docs_tool")
print("=" * 50)
result = query_docs_tool("What is Kubeflow Pipelines?")
print(f"Tool:  {result['tool']}")
print(f"Count: {result['count']}")
for r in result["results"]:
    print(f"\n  snippet : {r['golden_snippet'][:80]}")
    print(f"  url     : {r['validation_link']}")
    print(f"  score   : {r['relevance_score']}")

print("\n" + "=" * 50)
print("TEST 2 - query_code_tool")
print("=" * 50)
result = query_code_tool("webhook configuration YAML")
print(f"Tool:  {result['tool']}")
print(f"Count: {result['count']}")
for r in result["results"]:
    print(f"\n  snippet : {r['golden_snippet'][:80]}")
    print(f"  url     : {r['validation_link']}")
    print(f"  kind    : {r['kind']}")

print("\n" + "=" * 50)
print("TEST 3 - query_both_tool")
print("=" * 50)
result = query_both_tool("how to install and configure KServe")
print(f"Tool       : {result['tool']}")
print(f"Docs count : {len(result['docs_results'])}")
print(f"Code count : {len(result['code_results'])}")
print(f"Total      : {result['total_count']}")

print("\nALL MCP TOOL TESTS PASSED")