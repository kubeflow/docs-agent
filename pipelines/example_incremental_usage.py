#!/usr/bin/env python3
"""
Example script showing how to trigger the incremental RAG pipeline
with a list of changed files.

This script demonstrates:
1. How to format the changed files list
2. How to compile and run the incremental pipeline
3. Example integration with git to detect changed files
"""

import json
import subprocess
import kfp
from incremental_pipeline import github_rag_incremental_pipeline

def get_changed_files_from_git(repo_path=".", since_commit="HEAD~1"):
    """
    Get list of changed files from git diff.
    
    Args:
        repo_path: Path to git repository
        since_commit: Compare changes since this commit (default: last commit)
    
    Returns:
        List of changed file paths
    """
    try:
        # Get changed files between commits
        cmd = ["git", "diff", "--name-only", since_commit, "HEAD"]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
        
        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            # Filter for documentation files only
            doc_files = [f for f in files if f.endswith(('.md', '.html'))]
            return doc_files
        else:
            print(f"Git command failed: {result.stderr}")
            return []
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []


def run_incremental_pipeline_example():
    """
    Example of running the incremental pipeline with specific changed files.
    """
    
    # Example 1: Manually specified changed files
    changed_files = [
        "content/en/docs/started/getting-started.md",
        "content/en/docs/components/pipelines/overview.md",
        "content/en/docs/external-add-ons/kustomize.md"
    ]
    
    print("=== Manual File List Example ===")
    print(f"Changed files: {changed_files}")
    
    # Convert to JSON string (required by the pipeline)
    changed_files_json = json.dumps(changed_files)
    print(f"JSON format: {changed_files_json}")
    
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_incremental_pipeline,
        package_path="github_rag_incremental_pipeline.yaml"
    )
    print("✅ Pipeline compiled to: github_rag_incremental_pipeline.yaml")
    
    # Example pipeline parameters
    pipeline_params = {
        'repo_owner': 'kubeflow',
        'repo_name': 'website',
        'changed_files': changed_files_json,
        'github_token': '',  # Add your GitHub token here
        'base_url': 'https://www.kubeflow.org/docs',
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'milvus_host': 'milvus-standalone-final.santhosh.svc.cluster.local',
        'milvus_port': '19530',
        'collection_name': 'docs_rag'
    }
    
    print("\n=== Pipeline Parameters ===")
    for key, value in pipeline_params.items():
        if key == 'github_token' and value:
            print(f"{key}: {'*' * len(value)}")  # Hide token
        else:
            print(f"{key}: {value}")


def run_git_integration_example():
    """
    Example of integrating with git to automatically detect changed files.
    """
    print("\n=== Git Integration Example ===")
    
    # Get changed files from git
    changed_files = get_changed_files_from_git()
    
    if not changed_files:
        print("No documentation files changed since last commit.")
        return
    
    print(f"Detected {len(changed_files)} changed documentation files:")
    for file in changed_files:
        print(f"  - {file}")
    
    # Convert to JSON for pipeline
    changed_files_json = json.dumps(changed_files)
    
    print(f"\nJSON format for pipeline: {changed_files_json}")
    
    # You would trigger the pipeline here with changed_files_json
    print("\n📝 To run the pipeline with these files:")
    print("1. Set up your Kubeflow Pipelines client")
    print("2. Submit the pipeline with the changed_files parameter")
    print("3. Monitor the pipeline execution")


def webhook_trigger_example():
    """
    Example of how this could be triggered by a webhook from GitHub.
    """
    print("\n=== Webhook Integration Example ===")
    
    # Simulate webhook payload (GitHub push event)
    webhook_payload = {
        "commits": [
            {
                "modified": [
                    "content/en/docs/started/getting-started.md",
                    "content/en/docs/components/pipelines/overview.md"
                ],
                "added": [
                    "content/en/docs/new-feature.md"
                ],
                "removed": [
                    "content/en/docs/deprecated.md"
                ]
            }
        ]
    }
    
    # Extract all changed files
    all_changed_files = []
    for commit in webhook_payload["commits"]:
        all_changed_files.extend(commit.get("modified", []))
        all_changed_files.extend(commit.get("added", []))
        # Note: removed files would need special handling to delete from Milvus
    
    # Remove duplicates and filter for documentation files
    changed_files = list(set([f for f in all_changed_files if f.endswith(('.md', '.html'))]))
    
    print(f"Files to process from webhook: {changed_files}")
    
    if changed_files:
        changed_files_json = json.dumps(changed_files)
        print(f"Would trigger pipeline with: {changed_files_json}")
    else:
        print("No documentation files to process.")


def main():
    """
    Run all examples.
    """
    print("🚀 Incremental RAG Pipeline Examples")
    print("=" * 50)
    
    # Run examples
    run_incremental_pipeline_example()
    run_git_integration_example()
    webhook_trigger_example()
    
    print("\n" + "=" * 50)
    print("✅ Examples completed!")
    print("\n📚 Next steps:")
    print("1. Set up your GitHub token")
    print("2. Configure your Kubeflow Pipelines environment")
    print("3. Set up webhooks or CI/CD integration")
    print("4. Test with a small set of changed files")


if __name__ == "__main__":
    main()
