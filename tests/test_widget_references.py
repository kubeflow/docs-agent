"""Guard that the chat widget only calls functions it actually defines.

The widget is one plain script with no bundler or linter, and the only other
JS test loads the top-level prelude, so a call inside the DOMContentLoaded
closure to a name that no longer exists reaches the browser as a ReferenceError
with nothing in CI to catch it.
"""

import re
from pathlib import Path


CHATBOT_JS = Path(__file__).parent.parent / "frontend" / "docs_scripts" / "chatbot.js"

# Reserved words that are followed by `(` but are not calls.
KEYWORDS = {
    "async",
    "await",
    "case",
    "catch",
    "class",
    "delete",
    "do",
    "else",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "new",
    "of",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "typeof",
    "void",
    "while",
    "yield",
}

# Browser and language globals the widget legitimately calls.
GLOBALS = {
    "AbortController",
    "Blob",
    "Boolean",
    "Date",
    "Error",
    "Number",
    "Promise",
    "String",
    "TextDecoder",
    "URL",
    "alert",
    "atob",
    "confirm",
    "fetch",
    "setInterval",
    "setTimeout",
}


def blank_template_text(source: str) -> str:
    """Keep `${...}` interpolations from template literals, blank the markup.

    Interpolations are real code and are where the widget builds most of its
    DOM. The literal text around them is markup, and something like
    `Sources (${n})` would otherwise read as a call to `Sources`.
    """
    out = []
    index = 0
    while index < len(source):
        if source[index] != "`":
            out.append(source[index])
            index += 1
            continue

        index += 1  # opening backtick
        while index < len(source) and source[index] != "`":
            if source[index] == "\\":
                index += 2
                continue
            if source.startswith("${", index):
                depth = 1
                index += 2
                start = index
                while index < len(source) and depth:
                    if source[index] == "{":
                        depth += 1
                    elif source[index] == "}":
                        depth -= 1
                    index += 1
                end = index - 1 if depth == 0 else index
                out.append(" " + blank_template_text(source[start:end]) + " ")
                continue
            out.append("\n" if source[index] == "\n" else " ")
            index += 1
        index += 1  # closing backtick
    return "".join(out)


def strip_literals(source: str) -> str:
    """Blank comments, quoted strings and template markup."""
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    source = re.sub(r"(?<!:)//[^\n]*", " ", source)
    source = re.sub(r"'(?:\\.|[^'\\\n])*'", "''", source)
    source = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', source)
    return blank_template_text(source)


def called_names(source: str) -> set[str]:
    """Bare `name(` call sites, ignoring method calls and regex escapes."""
    found = re.findall(r"(?<![.\w$\\])([A-Za-z_$][\w$]*)\s*\(", source)
    return set(found) - KEYWORDS


def declared_names(source: str) -> set[str]:
    """Every name the file binds: declarations, classes and function parameters."""
    declared = set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", source))
    declared |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", source))
    declared |= set(re.findall(r"\bclass\s+([A-Za-z_$][\w$]*)", source))
    declared |= set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*=>", source))
    # Method shorthand in classes and object literals: `push(chunk) { ... }`.
    declared |= set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{", source))
    for params in re.findall(r"\((\s*[^()]*?)\)\s*(?:=>|\{)", source):
        for param in params.split(","):
            name = param.strip().split("=")[0].strip().lstrip(".")
            if re.fullmatch(r"[A-Za-z_$][\w$]*", name):
                declared.add(name)
    return declared


def unresolved_calls(source: str) -> set[str]:
    """Names the file calls but never binds.

    Two limits worth knowing. Quoted strings are blanked before template markup
    is, so an interpolation inside a quoted HTML attribute (`href="${fn(x)}"`)
    is not scanned; a single pass that handled both would also have to lex regex
    literals, which `/[&<>"\']/g` in this very file makes awkward. And names are
    resolved file-wide rather than per scope, so a parameter can mask a
    same-named call elsewhere.
    """
    stripped = strip_literals(source)
    return called_names(stripped) - declared_names(stripped) - GLOBALS


def test_widget_calls_only_functions_it_defines():
    source = CHATBOT_JS.read_text(encoding="utf-8")
    unresolved = unresolved_calls(source)

    assert not unresolved, (
        f"chatbot.js calls {sorted(unresolved)}, which nothing in the file declares. "
        "Either the helper was renamed and a call site was missed, or a new global "
        "belongs in GLOBALS."
    )


def test_the_check_would_catch_a_renamed_helper():
    """A rename that misses a call site must fail the test above."""
    source = CHATBOT_JS.read_text(encoding="utf-8").replace(
        "function formatChatMarkdown(", "function renamedFormatter("
    )

    assert "formatChatMarkdown" in unresolved_calls(source)
