import re

import frontmatter
from frontmatter.default_handlers import TOMLHandler, YAMLHandler
from html_table_rescuer import ParseConfig, RowspanStrategy, TableParser
from markdownify import markdownify as html_to_markdown

_MD_CONVERT_OPTS = {
    "heading_style": "ATX",
    "escape_asterisks": False,
    "escape_underscores": False,
    "strip": ["script", "style"],
}
_TABLE_PARSE_CFG = ParseConfig(
    rowspan_strategy=RowspanStrategy.REPEAT_VALUE,
    parser_library="html.parser",
)


def parse_frontmatter(content):
    """Return (metadata, body) for Hugo YAML (---) or TOML (+++) frontmatter."""
    if not content:
        return {}, content
    text = content.lstrip("\ufeff")
    handler = TOMLHandler() if text.startswith("+++") else YAMLHandler()
    try:
        meta, body = frontmatter.parse(text, handler=handler)
        return meta or {}, body
    except Exception:
        return {}, content


def process_html_table(html):
    """Turn HTML <table> nodes into Markdown tables; expand rowspan/colspan."""
    if "<table" not in html:
        return html
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        parsed = TableParser(str(table), config=_TABLE_PARSE_CFG).parse()
        if parsed:
            md = parsed[0].to_markdown()
        else:
            md = html_to_markdown(str(table), **_MD_CONVERT_OPTS)
        table.replace_with(soup.new_string("\n" + md.strip() + "\n"))
    return str(soup)


def clean_hugo_markdown(content):
    meta, body = parse_frontmatter(content)

    stashes = {}

    def stash_fence(m):
        k = f"%%FENCE{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k

    body = re.sub(r"```.*?```", stash_fence, body, flags=re.DOTALL)

    def stash_code(m):
        k = f"%%CODE{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k

    body = re.sub(r"`[^`\n]+`", stash_code, body)

    def stash_gfm(m):
        k = f"%%GFM{len(stashes)}%%"
        stashes[k] = m.group(0)
        return k

    body = re.sub(r"(?:\|.*\|[\r\n]+)+", stash_gfm, body)

    # Preserve placeholder tokens such as <YOUR_HF_TOKEN>; HTML parsers would
    # otherwise treat these uppercase values as tags.
    body = re.sub(r"<[A-Z][A-Z0-9_:-]*>", stash_code, body)

    body = re.sub(
        r"\{\{%\s*alert.*?%\}\}(.*?)\{\{%\s*/alert\s*%\}\}",
        r"NOTE: \1",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(r"\{\{.*?\}\}", "", body, flags=re.DOTALL)
    body = body.replace("fa-check", "yes").replace("fa-xmark", "no")

    body = html_to_markdown(body, **_MD_CONVERT_OPTS)
    body = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"Figure: \1", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"[ \t]+", " ", body)

    for k, v in stashes.items():
        body = body.replace(k, v)

    return meta, body
