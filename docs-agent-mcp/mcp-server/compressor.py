"""Extractive sentence compression for RAG hits."""

from __future__ import annotations

import re

import embeddings_client


def compress_hits(query: str, hits: list[dict], max_chars: int) -> list[dict]:
    """Compress hits extractively to fit within max_chars.

    Splits each hit's text into sentences, embeds them, scores them against
    the query embedding, and keeps only the highest-scoring sentences until
    the character limit is reached.
    """
    if not hits or max_chars <= 0:
        return hits

    try:
        query_embedding = embeddings_client.embed_query(query)
    except Exception as exc:
        print(f"Failed to embed query for compression: {exc}")
        return hits

    # Sentence boundary regex (basic approximation)
    # Matches punctuation followed by whitespace and a capital letter, or newlines
    sentence_splitter = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])|\n+")

    all_sentences = []
    # Keep track of which hit and which sentence index each sentence belongs to
    for hit_idx, hit in enumerate(hits):
        content = hit.get("entity", {}).get("content_text", "")
        if not content:
            continue

        sentences = [s.strip() for s in sentence_splitter.split(content) if s.strip()]
        for sent_idx, text in enumerate(sentences):
            all_sentences.append(
                {
                    "hit_idx": hit_idx,
                    "sent_idx": sent_idx,
                    "text": text,
                }
            )

    if not all_sentences:
        return hits

    # Embed all sentences
    texts = [s["text"] for s in all_sentences]
    try:
        sent_embeddings = embeddings_client.embed_texts(texts)
    except Exception as exc:
        print(f"Failed to embed sentences for compression: {exc}")
        return hits

    # Compute cosine similarity
    for s_info, s_emb in zip(all_sentences, sent_embeddings):
        # dot product for cosine similarity (assuming normalized vectors, which TEI all-mpnet usually does)
        # TEI all-mpnet-base-v2 returns normalized vectors.
        score = sum(a * b for a, b in zip(query_embedding, s_emb))
        s_info["score"] = score

    # Sort sentences globally by score descending
    all_sentences.sort(key=lambda x: x["score"], reverse=True)

    # Select sentences until budget is reached
    selected_sentences = []
    current_chars = 0
    for s_info in all_sentences:
        text_len = len(s_info["text"])
        if current_chars + text_len > max_chars and current_chars > 0:
            # We skip this sentence. If we haven't added anything yet, add at least one.
            continue
        selected_sentences.append(s_info)
        current_chars += text_len
        if current_chars >= max_chars:
            break

    # Group selected sentences back by hit
    hits_to_keep = {}
    for s_info in selected_sentences:
        h_idx = s_info["hit_idx"]
        if h_idx not in hits_to_keep:
            hits_to_keep[h_idx] = []
        hits_to_keep[h_idx].append(s_info)

    # Reconstruct the hits
    compressed_hits = []
    for h_idx in sorted(hits_to_keep.keys()):
        # Sort sentences in this hit back to their original document order
        hit_sentences = sorted(hits_to_keep[h_idx], key=lambda x: x["sent_idx"])
        combined_text = " ... ".join(s["text"] for s in hit_sentences)

        hit_copy = hits[h_idx].copy()
        hit_copy["entity"] = hit_copy["entity"].copy()
        hit_copy["entity"]["content_text"] = combined_text
        compressed_hits.append(hit_copy)

    print(
        f"Compressor: reduced {len(all_sentences)} sentences to {len(selected_sentences)} ({current_chars}/{max_chars} chars)"
    )
    return compressed_hits
