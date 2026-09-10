"""Tests for Memory.search() and Memory.views().

Most of these are metadata-only. The two that actually rank results download a
small sentence-transformers model but need no API key.
"""

import pytest

from pixelmemory import Memory
from pixelmemory.context import Document, Text

from conftest import EMBED_MODEL


def test_search_targets_registered(namespace):
    m = Memory(
        [Text(id="title", embed=False), Text(id="body", embed=True, embed_model=EMBED_MODEL)],
        namespace=namespace,
    )
    targets = m.search_targets
    assert [t.context_id for t in targets] == ["body"]
    # idx_name is recorded, not inferred: one column can carry two indexes.
    assert targets[0].index_name == "similarity"
    assert targets[0].column == "body"


def test_unembedded_memory_has_no_targets(namespace):
    m = Memory([Text(id="body", embed=False)], namespace=namespace)
    assert m.search_targets == []
    with pytest.raises(RuntimeError, match="no embedding index"):
        m.search("anything")


def test_search_on_unknown_context_lists_the_real_ones(namespace):
    m = Memory([Text(id="body", embed=True, embed_model=EMBED_MODEL)], namespace=namespace)
    with pytest.raises(KeyError) as excinfo:
        m.search("q", on="nope")
    assert "body" in str(excinfo.value)


def test_document_registers_its_chunk_view(namespace):
    """A chunked document is searched on the chunk view's text, not the column."""
    m = Memory([Document(id="report", embed=True, embed_model=EMBED_MODEL)], namespace=namespace)
    targets = {t.context_id: t for t in m.search_targets}
    assert "report" in targets
    assert targets["report"].column == "text"
    assert targets["report"].table.get_metadata()["name"].endswith("report_chunks")


def test_views_maps_context_id_to_table(namespace):
    """What the pre-0.2.0 examples reached for as memory.chunk_views["report"]."""
    m = Memory([Document(id="report", embed=True, embed_model=EMBED_MODEL)], namespace=namespace)
    views = m.views()
    assert "report" in views
    assert "text" in views["report"].columns()
    with pytest.raises(KeyError, match="report"):
        m.views("not_a_context")


def test_search_ranks_by_similarity(namespace):
    m = Memory(
        [Text(id="title", embed=False), Text(id="body", embed=True, embed_model=EMBED_MODEL)],
        namespace=namespace,
    )
    m.add(
        m.Entry(title="a", body="The cat sat on the mat."),
        m.Entry(title="b", body="Quantum chromodynamics describes the strong force."),
        m.Entry(title="c", body="A small kitten napped in the sun."),
    )
    hits = m.search("feline animal", limit=2)
    assert len(hits) == 2
    assert all(h["context_id"] == "body" for h in hits)
    assert "kitten" in hits[0]["text"] or "cat" in hits[0]["text"]
    assert hits[0]["score"] >= hits[1]["score"]
    assert "chromodynamics" not in " ".join(h["text"] for h in hits)


def test_search_min_score_filters(namespace):
    m = Memory([Text(id="body", embed=True, embed_model=EMBED_MODEL)], namespace=namespace)
    m.add(m.Entry(body="The cat sat on the mat."))
    assert m.search("cat", min_score=0.0) != []
    assert m.search("cat", min_score=1.1) == []
