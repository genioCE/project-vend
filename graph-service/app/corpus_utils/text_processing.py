"""Text processing utilities: markdown stripping and text chunking."""

from __future__ import annotations

import re


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, preserving the plain text content."""
    # Remove headers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove strikethrough
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # Remove inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove list markers
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 500) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.
    If a paragraph exceeds max_words, split at sentence boundaries.
    Never splits mid-sentence.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_words = len(para.split())

        # If adding this paragraph would exceed max_words and we have content,
        # flush the current chunk first
        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        # If a single paragraph exceeds max_words, split by sentences
        if para_words > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_chunk: list[str] = []
            sent_word_count = 0

            for sentence in sentences:
                s_words = len(sentence.split())
                if sent_word_count + s_words > max_words and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_word_count = 0
                sent_chunk.append(sentence)
                sent_word_count += s_words

            if sent_chunk:
                remaining = " ".join(sent_chunk)
                current_chunk.append(remaining)
                current_word_count += sent_word_count
        else:
            current_chunk.append(para)
            current_word_count += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if c.strip()]
