"""
Tests for sentence chunking functionality in PixelMemory.
"""

import pytest
import pixeltable as pxt
from pixelmemory import Memory


def test_sentence_chunking_basic():
    """Test basic sentence chunking functionality."""
    # Clean up any existing test data
    try:
        pxt.drop_dir("test_chunking", force=True)
    except:
        pass

    # Create Memory with sentence chunking enabled
    memory = Memory(
        namespace="test_chunking",
        table_name="test_docs",
        schema={
            "doc_id": pxt.Required[pxt.String],
            "content": pxt.String,
        },
        columns_to_index=["content"],
        use_sentence_chunking=True,
        primary_key="doc_id",
    )

    # Insert test document
    test_doc = [
        {
            "doc_id": "test1",
            "content": "This is the first sentence. This is the second sentence. This is the third sentence.",
        }
    ]

    memory.insert(test_doc)

    # Verify chunk view was created
    assert "content" in memory.chunk_views

    # Verify sentence chunks were created
    content_chunks = memory.chunk_views["content"]
    chunk_results = content_chunks.select(
        content_chunks.doc_id, content_chunks.text, content_chunks.pos
    ).collect()

    # Should have 3 sentence chunks
    assert len(chunk_results) == 3

    # Verify content of chunks
    expected_sentences = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
    ]

    for i, row in enumerate(chunk_results):
        assert row["doc_id"] == "test1"
        assert row["text"].strip() == expected_sentences[i]
        assert row["pos"] == i

    # Clean up
    pxt.drop_dir("test_chunking", force=True)


def test_sentence_chunking_disabled():
    """Test that sentence chunking is disabled by default."""
    try:
        pxt.drop_dir("test_no_chunking", force=True)
    except:
        pass

    # Create Memory without sentence chunking
    memory = Memory(
        namespace="test_no_chunking",
        table_name="test_docs",
        schema={
            "doc_id": pxt.Required[pxt.String],
            "content": pxt.String,
        },
        columns_to_index=["content"],
        use_sentence_chunking=False,  # Explicitly disabled
        primary_key="doc_id",
    )

    # Insert test document
    test_doc = [
        {
            "doc_id": "test1",
            "content": "This is the first sentence. This is the second sentence.",
        }
    ]

    memory.insert(test_doc)

    # Verify no chunk views were created for the content column
    assert "content" not in memory.chunk_views

    # Clean up
    pxt.drop_dir("test_no_chunking", force=True)


if __name__ == "__main__":
    test_sentence_chunking_basic()
    test_sentence_chunking_disabled()
    print("All tests passed!")
