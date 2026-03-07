from app.ingest import chunk_text, parse_date_from_filename, strip_markdown


def test_parse_date_from_filename_supports_multiple_formats():
    assert parse_date_from_filename("1-2-2025.md") == "2025-01-02"
    assert parse_date_from_filename("2025-08-18.md") == "2025-08-18"
    assert parse_date_from_filename("20250818.md") == "2025-08-18"
    assert parse_date_from_filename("notes.md") is None


def test_strip_markdown_preserves_plain_text_content():
    raw = "# Header\n\nThis is **bold** and [a link](https://example.com)."
    cleaned = strip_markdown(raw)
    assert "Header" in cleaned
    assert "**" not in cleaned
    assert "a link" in cleaned
    assert "https://example.com" not in cleaned


def test_chunk_text_respects_word_boundaries():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_text(text, max_words=4)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 4 for c in chunks)
