"""Structure-aware, extractive sentence & block compressor for RAG hits."""

from __future__ import annotations

import re

# Common English stop words to ignore during lexical term matching
STOP_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
}

# Regex to detect fenced code blocks (```...```)
CODE_BLOCK_RE = re.compile(r"(```[a-zA-Z0-9_\-\.]*\n[\s\S]*?\n```)", re.MULTILINE)

# Sentence boundary regex
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"`\'])')
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "fig.", "v1.", "v2.", "vol.", "no.")


BLOCK_START_RE = re.compile(r"^(\s*([-*+]|\d+\.)\s|#{1,6}\s|\||>)")
LIST_OR_HEADING_RE = re.compile(r"^(\s*([-*+]|\d+\.)\s|#{1,6}\s)")


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s", line))


def _logical_lines(para: str) -> list[str]:
    """Merge hard-wrapped lines; a new line starts only at a list item, heading, table row or quote."""
    out: list[str] = []
    for line in para.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if out and not _is_heading(out[-1]) and not BLOCK_START_RE.match(line):
            out[-1] = f"{out[-1]} {stripped}"
        else:
            out.append(stripped)
    return out


def _split_into_blocks(text: str) -> list[str]:
    """Split text into atomic blocks (fenced code blocks, tables, list items, sentences)."""
    if not text:
        return []

    # 1. Separate code blocks from non-code text
    segments: list[tuple[bool, str]] = []
    last_end = 0
    for match in CODE_BLOCK_RE.finditer(text):
        start, end = match.span()
        if start > last_end:
            segments.append((False, text[last_end:start]))
        segments.append((True, match.group(0)))
        last_end = end
    if last_end < len(text):
        segments.append((False, text[last_end:]))

    blocks: list[str] = []
    for is_code, segment in segments:
        if is_code:
            stripped = segment.strip()
            if stripped:
                blocks.append(stripped)
            continue

        paragraphs = re.split(r"\n\s*\n+", segment)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect Markdown tables as single atomic blocks
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            if len(lines) >= 2 and all(line.startswith("|") and line.endswith("|") for line in lines):
                blocks.append(para)
                continue

            # Merge hard-wrapped lines into logical lines
            logical_lines = _logical_lines(para)
            for line in logical_lines:
                # Keep list items (- **key**: ...) and headings whole as single blocks
                if LIST_OR_HEADING_RE.match(line):
                    blocks.append(line)
                    continue

                # Plain prose lines: split into sentences
                raw_sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(line) if s.strip()]
                sentences: list[str] = []
                for s in raw_sentences:
                    if sentences and any(sentences[-1].lower().endswith(abbr) for abbr in ABBREVIATIONS):
                        sentences[-1] = f"{sentences[-1]} {s}"
                    else:
                        sentences.append(s)

                blocks.extend(sentences)

    return blocks


def _extract_query_terms(query: str) -> list[str]:
    """Extract informative non-stopword tokens from the query."""
    tokens = re.findall(r"[A-Za-z0-9_\-\.]+", query.lower())
    informative = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return informative or tokens


def _score_block(block: str, query: str, query_terms: list[str], is_lead: bool = False) -> float:
    """Score a block using in-memory lexical matching and structural heuristics."""
    block_lower = block.lower()
    score = 0.0

    # 1. Lead sentence context anchoring bonus
    if is_lead:
        score += 1.5

    # 2. Exact full query phrase match bonus
    q_clean = query.lower().strip()
    if len(q_clean) > 3 and q_clean in block_lower:
        score += 5.0

    # 3. Informative query term matches
    term_hits = 0
    for term in query_terms:
        if term in block_lower:
            term_hits += 1
            if re.search(rf"\b{re.escape(term)}\b", block_lower):
                score += 2.0
            else:
                score += 1.0

    if query_terms:
        score += (term_hits / len(query_terms)) * 3.0

    # 4. Code block bonus if query seeks examples/manifests/config
    is_code = block.startswith("```") and block.endswith("```")
    if is_code:
        code_intent_words = {"example", "yaml", "manifest", "config", "spec", "code", "command", "args"}
        if any(w in q_clean for w in code_intent_words) or term_hits > 0:
            score += 2.5

    return score


def _allocate_budgets(hits: list[dict], max_chars: int) -> list[int]:
    """Allocate character budget per hit based on rank decay weights."""
    n = len(hits)
    if n == 0:
        return []
    if n == 1:
        return [max_chars]

    weights = [1.0 / ((i + 1) ** 0.6) for i in range(n)]
    total_weight = sum(weights)
    budgets = [max(100, int((w / total_weight) * max_chars)) for w in weights]

    diff = max_chars - sum(budgets)
    budgets[0] += diff
    return budgets


def compress_hits(query: str, hits: list[dict], max_chars: int) -> list[dict]:
    """Compress hits extractively to fit within max_chars.

    Uses structure-aware segmentation (protecting code blocks and tables),
    lead-sentence anchoring, in-memory lexical scoring, and rank-weighted
    budget allocation to prevent document starvation and eliminate network latency.
    """
    if not hits or max_chars <= 0:
        return hits

    # Early exit pass-through if all hits already fit within budget
    total_chars = sum(len(hit.get("entity", {}).get("content_text", "")) for hit in hits)
    if total_chars <= max_chars:
        return hits

    query_terms = _extract_query_terms(query)
    budgets = _allocate_budgets(hits, max_chars)

    compressed_hits = []
    surplus_budget = 0

    for hit_idx, hit in enumerate(hits):
        content = hit.get("entity", {}).get("content_text", "")
        if not content:
            continue

        allocated_budget = budgets[hit_idx] + surplus_budget
        surplus_budget = 0

        # If document already fits inside allocated budget, keep it intact
        if len(content) <= allocated_budget:
            compressed_hits.append(hit)
            surplus_budget = allocated_budget - len(content)
            continue

        blocks = _split_into_blocks(content)
        if not blocks:
            compressed_hits.append(hit)
            continue

        scored_blocks = []
        for idx, block in enumerate(blocks):
            is_lead = idx == 0
            score = _score_block(block, query, query_terms, is_lead=is_lead)
            scored_blocks.append({"idx": idx, "text": block, "score": score, "len": len(block)})

        scored_blocks_sorted = sorted(scored_blocks, key=lambda x: x["score"], reverse=True)

        selected = []
        used_chars = 0

        lead_block = scored_blocks[0]
        if lead_block["len"] <= allocated_budget:
            selected.append(lead_block)
            used_chars += lead_block["len"]

        for b in scored_blocks_sorted:
            if b["idx"] == 0 and lead_block in selected:
                continue
            if used_chars + b["len"] > allocated_budget:
                if not selected:
                    selected.append(b)
                    used_chars += b["len"]
                continue
            selected.append(b)
            used_chars += b["len"]

        selected.sort(key=lambda x: x["idx"])
        surplus_budget = max(0, allocated_budget - used_chars)

        parts = [b["text"] for b in selected]
        combined_text = "\n\n".join(parts) if any(p.startswith("```") for p in parts) else " ".join(parts)

        hit_copy = hit.copy()
        hit_copy["entity"] = hit_copy["entity"].copy()
        hit_copy["entity"]["content_text"] = combined_text
        compressed_hits.append(hit_copy)

    return compressed_hits
