"""
Code Ingestion — Repo Cloner Component

Clones the kubeflow/manifests repository and collects file metadata.
Records commit SHA for provenance tracking.

Features:
  - Clones via subprocess (git) or GitPython
  - Skips hidden dirs, __pycache__, node_modules
  - Size filter: skip files < 200 bytes or > 100KB
  - Groups files by extension

Environment variables:
  KUBEFLOW_MANIFESTS_REPO: Repo URL (default: https://github.com/kubeflow/manifests)
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".tox", ".mypy_cache",
    ".pytest_cache", ".venv", "venv", ".eggs", "*.egg-info",
}

SUPPORTED_EXTENSIONS = {".py", ".go", ".yaml", ".yml", ".md"}

MIN_FILE_SIZE = 200       # bytes
MAX_FILE_SIZE = 100_000   # 100KB


def get_repo_url() -> str:
    """Get the repository URL from environment.

    Returns:
        Repository URL string.
    """
    return os.environ.get(
        "KUBEFLOW_MANIFESTS_REPO",
        "https://github.com/kubeflow/manifests",
    )


def clone_repo(
    repo_url: Optional[str] = None,
    target_dir: Optional[str] = None,
    branch: str = "master",
) -> Dict[str, Any]:
    """Clone a git repository and collect file metadata.

    Args:
        repo_url: Repository URL to clone.
        target_dir: Directory to clone into (temp dir if None).
        branch: Git branch to clone.

    Returns:
        Dict with commit_sha, repo_dir, and file_list.
    """
    url = repo_url or get_repo_url()
    clone_dir = target_dir or tempfile.mkdtemp(prefix="docs-agent-code-")

    logger.info("Cloning %s (branch: %s) to %s", url, branch, clone_dir)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, url, clone_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Git clone failed: %s", e.stderr)
        raise RuntimeError(f"Failed to clone {url}: {e.stderr}") from e

    # Get commit SHA
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=clone_dir, check=True,
        )
        commit_sha = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit_sha = "unknown"

    logger.info("Cloned at commit: %s", commit_sha[:12])

    # Walk and collect files
    file_list = collect_files(clone_dir)

    return {
        "commit_sha": commit_sha,
        "repo_dir": clone_dir,
        "file_list": file_list,
    }


def should_skip_dir(dir_name: str) -> bool:
    """Check if a directory should be skipped.

    Args:
        dir_name: Directory basename.

    Returns:
        True if the directory should be skipped.
    """
    if dir_name.startswith("."):
        return True
    return dir_name in SKIP_DIRS


def collect_files(repo_dir: str) -> List[Dict[str, Any]]:
    """Walk a directory and collect file metadata.

    Filters by extension, size, and skips hidden/utility directories.

    Args:
        repo_dir: Root directory to walk.

    Returns:
        List of file info dicts: {path, extension, size_bytes, folder_context}.
    """
    files = []

    for root, dirs, filenames in os.walk(repo_dir):
        # Filter out directories to skip (modifies in-place)
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, repo_dir)

            # Check extension
            _, ext = os.path.splitext(filename)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Check size
            try:
                size = os.path.getsize(filepath)
            except OSError:
                continue

            if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                continue

            # Determine folder context (top-level directory)
            parts = rel_path.split(os.sep)
            folder_context = parts[0] if len(parts) > 1 else "root"

            files.append({
                "path": rel_path,
                "extension": ext.lower(),
                "size_bytes": size,
                "folder_context": folder_context,
            })

    # Log summary by extension
    ext_counts: Dict[str, int] = {}
    for f in files:
        ext_counts[f["extension"]] = ext_counts.get(f["extension"], 0) + 1

    logger.info("Collected %d files: %s", len(files), ext_counts)
    return files


def read_file_content(repo_dir: str, file_path: str) -> Optional[str]:
    """Read file content safely.

    Args:
        repo_dir: Repository root directory.
        file_path: Relative file path.

    Returns:
        File content string, or None if unreadable.
    """
    full_path = os.path.join(repo_dir, file_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logger.warning("Cannot read %s: %s", file_path, e)
        return None


def save_clone_results(
    result: Dict[str, Any], output_path: str
) -> None:
    """Save clone results to a JSON file.

    Args:
        result: Clone result dict.
        output_path: Path to write the file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Don't include repo_dir in the saved output (it's a temp path)
    save_data = {
        "commit_sha": result["commit_sha"],
        "file_count": len(result["file_list"]),
        "file_list": result["file_list"],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Saved clone results to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("=== Repo Cloner Smoke Test ===")
    result = clone_repo()
    logger.info("Commit: %s", result["commit_sha"][:12])
    logger.info("Files: %d", len(result["file_list"]))
    for f in result["file_list"][:10]:
        logger.info("  %s (%s, %d bytes)", f["path"], f["extension"], f["size_bytes"])
    # Cleanup
    shutil.rmtree(result["repo_dir"], ignore_errors=True)
