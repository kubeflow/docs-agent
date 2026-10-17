"""The MCP image copies an explicit file list; every local module the server imports must be in it.

flo_tracing.py was added without updating the list, and every image built from main
crashed on start with ModuleNotFoundError until this test existed.
"""

import ast
import re
from pathlib import Path

SERVER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"


def copied_files() -> set[str]:
    files = set()
    for line in (SERVER_DIR / "Dockerfile").read_text().splitlines():
        m = re.match(r"\s*COPY\s+(.+)\s+\S+\s*$", line)
        if m:
            files.update(m.group(1).split())
    return files


def local_imports(path: Path, local: set[str]) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names & local


def test_every_imported_local_module_is_copied_into_the_image():
    local = {p.stem for p in SERVER_DIR.glob("*.py")}
    copied = copied_files()
    missing = set()
    for name in copied:
        if name.endswith(".py"):
            missing |= {f"{m}.py" for m in local_imports(SERVER_DIR / name, local)} - copied
    assert "server.py" in copied
    assert not missing, f"imported but not copied by the Dockerfile: {sorted(missing)}"
