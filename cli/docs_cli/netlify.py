"""Netlify & kubeflow/website Hugo + Docsy integration for docs-cli."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

from .config import DocsConfig

console = Console()

MARKER_BEGIN = "<!-- BEGIN KUBEFLOW DOCS AGENT -->"
MARKER_END = "<!-- END KUBEFLOW DOCS AGENT -->"


def build_agent_endpoint_url(domain: str, scheme: str = "https") -> str:
    """Return the full A2A endpoint URL expected by chatbot.js."""
    clean = domain.strip().rstrip("/")
    if clean.startswith("http://") or clean.startswith("https://"):
        base = clean
    else:
        base = f"{scheme}://{clean}"
    return f"{base}/a2a/docs-agent/kubeflow-docs-agent"


def _upsert_marked_snippet(target_file: Path, snippet: str) -> None:
    """Insert or replace a marker-delimited block in an HTML partial file."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    block = f"{MARKER_BEGIN}\n{snippet.strip()}\n{MARKER_END}\n"
    if not target_file.exists():
        target_file.write_text(block, encoding="utf-8")
        return

    existing = target_file.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(MARKER_BEGIN)}.*?{re.escape(MARKER_END)}\n?",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(block, existing)
    else:
        sep = "\n" if existing and not existing.endswith("\n") else ""
        updated = f"{existing}{sep}{block}"
    target_file.write_text(updated, encoding="utf-8")


def inject_hugo_docsy_hooks(
    cfg: DocsConfig,
    domain: str,
    website_dir: Optional[Path] = None,
    scheme: str = "https",
) -> Dict[str, Path]:
    """Inject docs-agent chatbot assets and Docsy hooks into a kubeflow/website checkout.

    How kubeflow/website (Hugo + Docsy) maps to docs-agent:
    - `layouts/partials/head.html` includes `{{ partial "hooks/head-end.html" . -}}`
    - `layouts/partials/scripts.html` includes `{{ partial "hooks/body-end.html" . -}}`
    - `static/docs_styles/chatbot.css` -> published at `/docs_styles/chatbot.css`
    - `static/docs_scripts/chatbot.js` -> published at `/docs_scripts/chatbot.js`
    """
    target_dir = (
        website_dir.resolve()
        if website_dir
        else (cfg.repo_root / cfg.netlify.website_repo_path).resolve()
    )
    if not target_dir.exists():
        raise FileNotFoundError(
            f"kubeflow/website directory not found at {target_dir}. "
            "Set KUBEFLOW_WEBSITE_PATH or [netlify].website_repo_path in docs.toml."
        )

    agent_url = build_agent_endpoint_url(domain, scheme=scheme)

    src_css = cfg.frontend_dir / "docs_styles" / "chatbot.css"
    src_js = cfg.frontend_dir / "docs_scripts" / "chatbot.js"
    if not src_css.exists() or not src_js.exists():
        raise FileNotFoundError(
            f"Missing frontend chatbot assets under {cfg.frontend_dir}"
        )

    dst_css = target_dir / "static" / "docs_styles" / "chatbot.css"
    dst_js = target_dir / "static" / "docs_scripts" / "chatbot.js"
    dst_css.parent.mkdir(parents=True, exist_ok=True)
    dst_js.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_css, dst_css)
    shutil.copy2(src_js, dst_js)

    head_hook = target_dir / "layouts" / "partials" / "hooks" / "head-end.html"
    body_hook = target_dir / "layouts" / "partials" / "hooks" / "body-end.html"

    _upsert_marked_snippet(
        head_hook,
        '<link rel="stylesheet" href="/docs_styles/chatbot.css">',
    )
    _upsert_marked_snippet(
        body_hook,
        f"""<script>
  window.KUBEFLOW_DOCS_AGENT_URL = {json.dumps(agent_url)};
</script>
<script src="/docs_scripts/chatbot.js" data-agent-url="{agent_url}" defer></script>""",
    )

    return {
        "website_dir": target_dir,
        "css": dst_css,
        "js": dst_js,
        "head_hook": head_hook,
        "body_hook": body_hook,
    }


def prepare_demo_bundle(
    cfg: DocsConfig,
    domain: str,
    out_dir: Optional[Path] = None,
    scheme: str = "https",
) -> Path:
    """Build a standalone static site bundle from frontend/ in .docs-cli/netlify-dist/."""
    dist_dir = out_dir.resolve() if out_dir else (cfg.state_dir / "netlify-dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    agent_url = build_agent_endpoint_url(domain, scheme=scheme)

    # Copy static assets from frontend/
    shutil.copytree(
        cfg.frontend_dir / "docs_styles", dist_dir / "docs_styles", dirs_exist_ok=True
    )
    shutil.copytree(
        cfg.frontend_dir / "docs_scripts", dist_dir / "docs_scripts", dirs_exist_ok=True
    )

    kf_dash_url = "http://localhost:8085/"
    kf_pipe_url = "http://localhost:8085/pipeline/"
    kf_models_url = "http://localhost:8085/kserve-endpoints/"
    nav_banner = f"""<div style="background:#0f172a;color:#f8fafc;padding:10px 20px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #334155;z-index:9999;position:relative;">
  <div><strong>Kubeflow Docs Agent Cluster</strong> <span style="color:#94a3b8;margin-left:8px;">Namespaces: <code>kubeflow</code> | <code>ml-infra</code> | <code>docs-agent</code></span></div>
  <div style="display:flex;gap:12px;">
    <a href="{kf_dash_url}" target="_blank" style="color:#38bdf8;text-decoration:none;font-weight:600;">Kubeflow Central Dashboard ↗</a>
    <a href="{kf_pipe_url}" target="_blank" style="color:#a7f3d0;text-decoration:none;font-weight:600;">KFP Pipelines (kubeflow) ↗</a>
    <a href="{kf_models_url}" target="_blank" style="color:#fde68a;text-decoration:none;font-weight:600;">KServe Models (ml-infra) ↗</a>
  </div>
</div>"""

    index_src = cfg.frontend_dir / "index.html"
    if index_src.exists():
        index_html = index_src.read_text(encoding="utf-8")
        if "data-agent-url=" in index_html:
            index_html = re.sub(
                r'data-agent-url="[^"]*"',
                f'data-agent-url="{agent_url}"',
                index_html,
            )
        else:
            index_html = index_html.replace(
                '<script src="docs_scripts/chatbot.js"',
                f'<script>window.KUBEFLOW_DOCS_AGENT_URL = {json.dumps(agent_url)};</script>\n    <script src="docs_scripts/chatbot.js" data-agent-url="{agent_url}"',
            )
        if "<body>" in index_html and "Kubeflow Central Dashboard ↗" not in index_html:
            index_html = index_html.replace("<body>", f"<body>\n{nav_banner}", 1)
    else:
        index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kubeflow Docs Agent — Dev Cluster Preview</title>
  <link rel="stylesheet" href="/docs_styles/chatbot.css" />
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 3rem 2rem;
      background: #0f172a;
      color: #f8fafc;
    }}
    .card {{
      max-width: 720px;
      margin: 0 auto;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 2rem;
    }}
    code {{
      background: #0f172a;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      color: #38bdf8;
    }}
  </style>
</head>
<body>
{nav_banner}
  <div class="card" style="margin-top:2rem;">
    <h1>Kubeflow Docs Agent — Dev Preview</h1>
    <p>Connected to OKE Kagent A2A endpoint:</p>
    <p><code id="active-agent-endpoint">{agent_url}</code></p>
    <p>Use the chat launcher in the bottom-right corner to test Docs, Think, and Debug personas.</p>
  </div>
  <script>
    (function() {{
      try {{
        var raw = localStorage.getItem("chatbot_chats");
        if (raw && raw.indexOf("Connection error.") !== -1) {{
          localStorage.removeItem("chatbot_chats");
          localStorage.removeItem("chatbot_current_chat_id");
        }}
      }} catch (e) {{}}
      var h = window.location.hostname;
      if (h === "localhost" || h === "127.0.0.1" || h.endsWith(".nip.io")) {{
        window.KUBEFLOW_DOCS_AGENT_URL = window.location.origin + "/a2a/docs-agent/kubeflow-docs-agent";
      }} else {{
        window.KUBEFLOW_DOCS_AGENT_URL = {json.dumps(agent_url)};
      }}
      var el = document.getElementById("active-agent-endpoint");
      if (el) el.textContent = window.KUBEFLOW_DOCS_AGENT_URL;
    }})();
  </script>
  <script src="/docs_scripts/chatbot.js" data-agent-url="{agent_url}" defer></script>
</body>
</html>
"""

    (dist_dir / "index.html").write_text(index_html, encoding="utf-8")

    clean_domain = domain.strip().rstrip("/")
    if not clean_domain.startswith("http"):
        clean_domain = f"{scheme}://{clean_domain}"
    netlify_toml = f"""[build]
  publish = "."

[[redirects]]
  from = "/api/session"
  to = "{clean_domain}/api/session"
  status = 200
  force = true

[[redirects]]
  from = "/api/feedback"
  to = "{clean_domain}/api/feedback"
  status = 200
  force = true

[[redirects]]
  from = "/a2a/*"
  to = "{clean_domain}/a2a/:splat"
  status = 200
  force = true

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
"""
    (dist_dir / "netlify.toml").write_text(netlify_toml, encoding="utf-8")
    return dist_dir


def build_hugo_site(website_dir: Path, base_url: Optional[str] = None) -> Path:
    """Run `hugo --gc --minify` inside website_dir and return website_dir/public."""
    cmd = ["hugo", "--gc", "--minify"]
    if base_url:
        cmd.extend(["--baseURL", base_url])
    console.print(f"[dim]$ {' '.join(cmd)} (cwd={website_dir})[/dim]")
    subprocess.run(cmd, cwd=str(website_dir), check=True, text=True)
    return website_dir / "public"


def deploy_directory_to_netlify_api(
    publish_dir: Path,
    site_id: str,
    auth_token: str,
    prod: bool = True,
) -> Dict[str, Any]:
    """Deploy a static directory to Netlify using the native `netlify` CLI if available, falling back to the REST API."""
    if not site_id or not auth_token:
        raise ValueError(
            "Both NETLIFY_SITE_ID and NETLIFY_AUTH_TOKEN are required for Netlify deployment."
        )

    netlify_bin = shutil.which("netlify")
    if netlify_bin:
        cmd = [
            netlify_bin,
            "deploy",
            f"--dir={publish_dir}",
            f"--site={site_id}",
            f"--auth={auth_token}",
            "--json",
        ]
        if prod:
            cmd.append("--prod")
        console.print(f"[dim]$ netlify deploy --dir={publish_dir} --site={site_id} {'--prod' if prod else ''} --json[/dim]")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in publish_dir.rglob("*"):
            if file_path.is_file():
                rel_name = file_path.relative_to(publish_dir).as_posix()
                zf.write(file_path, arcname=rel_name)
    zip_bytes = buf.getvalue()

    url = f"https://api.netlify.com/api/v1/sites/{site_id}/deploys"
    req = urllib.request.Request(
        url,
        data=zip_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/zip",
            "User-Agent": "kubeflow-docs-cli/0.1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)
