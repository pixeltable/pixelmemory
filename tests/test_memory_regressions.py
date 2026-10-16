"""Exercise persistence and search with a deterministic, offline embedding UDF."""

import os
import subprocess
import sys

import numpy as np
import pixeltable as pxt
import pytest

from pixelmemory import Memory
from pixelmemory.context import Audio, Document, Image, Text, Video


@pxt.udf
def regression_embedding(text: str) -> pxt.Array[(3,), pxt.Float]:
    if text == "embedding failure":
        raise ValueError("intentional embedding failure")
    return np.array([1.0, float("cat" in text), float("dog" in text)], dtype=np.float32)


def test_multiple_embedded_columns_are_searchable(namespace):
    memory = Memory(
        [
            Text(id="title", embed_model=regression_embedding),
            Text(id="body", embed_model=regression_embedding),
        ],
        namespace=namespace,
    )
    memory.add(memory.Entry(title="cat title", body="dog body"))
    assert {hit["context_id"] for hit in memory.search("cat")} == {"title", "body"}
    assert len(memory.table.get_metadata()["indexes"]) == 2


def test_multiple_chunked_columns_have_distinct_direct_indexes(namespace):
    memory = Memory(
        [
            Text(id="title", use_chunking=True, embed_model=regression_embedding),
            Text(id="body", use_chunking=True, embed_model=regression_embedding),
        ],
        namespace=namespace,
    )
    indexes = memory.table.get_metadata()["indexes"]
    assert {tuple(index["columns"]) for index in indexes.values()} == {
        ("title",),
        ("body",),
    }
    assert len(memory.search_targets) == 4


@pytest.mark.parametrize("kind", [Text, Image, Document])
def test_optional_contexts_are_indexed(kind, namespace):
    memory = Memory(
        [kind(id="optional", required=False, embed_model=regression_embedding)],
        namespace=namespace,
    )
    assert [target.context_id for target in memory.search_targets] == ["optional"]


def test_optional_entry_and_null_scores(namespace):
    memory = Memory(
        [
            Text(id="note", required=False, embed_model=regression_embedding),
            Text(id="body", embed=False),
        ],
        namespace=namespace,
    )
    memory.add(memory.Entry("missing"), memory.Entry("present", note="cat"))
    hits = memory.search("cat", limit=1)
    assert len(hits) == 1
    assert hits[0]["text"] == "cat"
    assert memory.search("cat", min_score=0.0) == hits


def test_trailing_optional_fields_keep_positional_arguments(namespace):
    memory = Memory(
        [Text(id="body", embed=False), Text(id="note", required=False, embed=False)],
        namespace=namespace,
    )
    assert memory.Entry("cat").note is None
    assert memory.Entry("cat", "dog").note == "dog"


def test_failed_embeddings_do_not_hide_valid_matches(namespace):
    memory = Memory(
        [Text(id="body", embed_model=regression_embedding)],
        namespace=namespace,
        on_error="ignore",
    )
    memory.table.insert(
        [{"body": "embedding failure"}, {"body": "cat"}], on_error="ignore"
    )
    assert [hit["text"] for hit in memory.search("cat", limit=1)] == ["cat"]


def test_legacy_indexes_are_reused_without_reembedding(namespace):
    pxt.create_dir(namespace)
    table = pxt.create_table(f"{namespace}.memory", {"body": pxt.String})
    table.add_embedding_index(
        "body", embedding=regression_embedding, idx_name="similarity"
    )
    table.insert(body="cat")
    version = table.get_metadata()["version"]
    memory = Memory(
        [Text(id="body", embed_model=regression_embedding)], namespace=namespace
    )
    assert table.get_metadata()["version"] == version
    assert memory.search_targets[0].index_name == "similarity"
    assert memory.search("cat")[0]["text"] == "cat"


def test_legacy_collision_is_repaired_without_replacing_existing_index(namespace):
    pxt.create_dir(namespace)
    table = pxt.create_table(
        f"{namespace}.memory", {"title": pxt.String, "body": pxt.String}
    )
    table.add_embedding_index(
        "title", embedding=regression_embedding, idx_name="similarity"
    )
    memory = Memory(
        [
            Text(id="title", embed_model=regression_embedding),
            Text(id="body", embed_model=regression_embedding),
        ],
        namespace=namespace,
    )
    assert memory.search_targets[0].index_name == "similarity"
    assert memory.search_targets[1].index_name == "body_similarity"
    assert table.get_metadata()["indexes"]["similarity"]["columns"] == ["title"]


def test_explicit_index_name_collision_is_rejected(namespace):
    with pytest.raises(ValueError, match="already belongs to another column"):
        Memory(
            [
                Text(id="title", index_name="shared", embed_model=regression_embedding),
                Text(id="body", index_name="shared", embed_model=regression_embedding),
            ],
            namespace=namespace,
        )


def test_index_name_collisions_are_case_insensitive(namespace):
    with pytest.raises(ValueError, match="already belongs to another column"):
        Memory(
            [
                Text(id="title", index_name="Shared", embed_model=regression_embedding),
                Text(id="body", index_name="shared", embed_model=regression_embedding),
            ],
            namespace=namespace,
        )


@pytest.mark.parametrize(
    "context, conflicting_id",
    [
        (Image(id="photo"), "photo_response"),
        (Image(id="photo"), "photo_description"),
        (Video(id="clip"), "clip_audio"),
    ],
    ids=["response", "description", "audio"],
)
def test_generated_column_collisions_fail_before_mutation(
    context, conflicting_id, namespace
):
    with pytest.raises(ValueError, match="conflicts with a generated column"):
        Memory([context, Text(id=conflicting_id, embed=False)], namespace=namespace)
    assert namespace not in pxt.list_dirs()


@pytest.mark.parametrize(
    "contexts",
    [
        [Text(id="frame_response", embed=False), Video(id="clip")],
        [Text(id="frame_description", embed=False), Video(id="clip")],
        [Image(id="frame"), Video(id="clip")],
        [Video(id="clip"), Image(id="frame")],
        [Text(id="aud_transcription", embed=False), Audio(id="aud")],
        [Text(id="clip_transcription", embed=False), Video(id="clip")],
    ],
    ids=[
        "frame_response",
        "frame_description",
        "image_video",
        "video_image",
        "audio",
        "video",
    ],
)
def test_generated_view_collisions_fail_before_mutation(contexts, namespace):
    with pytest.raises(ValueError, match="conflicts with a generated view column"):
        Memory(contexts, namespace=namespace)
    assert namespace not in pxt.list_dirs()


def test_disabled_video_pipeline_does_not_reserve_view_columns(namespace):
    memory = Memory(
        [Text(id="frame_description", embed=False), Video(id="clip", embed=False)],
        namespace=namespace,
    )
    assert memory.table.columns() == ["frame_description", "clip"]


def test_repeated_setup_is_idempotent_and_empty_selection_is_noop(namespace):
    memory = Memory(
        [Text(id="body", embed_model=regression_embedding)], namespace=namespace
    )
    memory.add(memory.Entry("cat"))
    version = memory.table.get_metadata()["version"]
    memory.setup_indexing([])
    memory.setup_indexing()
    assert len(memory.search_targets) == 1
    assert memory.table.get_metadata()["version"] == version
    assert len(memory.search("cat")) == 1


@pytest.mark.parametrize("name", ["missing", "plain"])
def test_setup_rejects_unknown_or_unembedded_columns(name, namespace):
    memory = Memory([Text(id="plain", embed=False)], namespace=namespace)
    with pytest.raises(ValueError, match="unknown or have embed=False"):
        memory.setup_indexing([name])


@pytest.mark.parametrize("ids", [["body", "body"], ["body", "BODY"]])
def test_duplicate_context_ids_fail_before_catalog_mutation(ids, namespace):
    with pytest.raises(ValueError, match="Duplicate context id"):
        Memory([Text(id=name, embed=False) for name in ids], namespace=namespace)
    assert namespace not in pxt.list_dirs()


@pytest.mark.parametrize("context_id", ["count", "_rowid"])
def test_reserved_context_ids_fail_before_catalog_mutation(context_id, namespace):
    with pytest.raises(pxt.Error, match="reserved name|Invalid column name"):
        Memory([Text(id=context_id, embed=False)], namespace=namespace)
    assert namespace not in pxt.list_dirs()


def test_mixed_case_context_reopens(namespace):
    contexts = [Text(id="Body", embed=False)]
    memory = Memory(contexts, namespace=namespace)
    memory.add(memory.Entry(Body="cat"))
    reopened = Memory(contexts, namespace=namespace)
    assert reopened.count() == 1


@pytest.mark.parametrize(
    "changed",
    [
        [Text(id="renamed", embed=False)],
        [Text(id="body", embed=False, required=False)],
        [Image(id="body", embed=False)],
    ],
)
def test_reopen_rejects_changed_schema_without_losing_rows(changed, namespace):
    memory = Memory([Text(id="body", embed=False)], namespace=namespace)
    memory.add(memory.Entry("cat"))
    with pytest.raises(ValueError, match="different context schema"):
        Memory(changed, namespace=namespace)
    assert memory.count() == 1


def test_crud_and_reopen_from_another_process(namespace):
    memory = Memory([Text(id="body", embed=False)], namespace=namespace)
    memory.add(memory.Entry("cat"), memory.Entry("dog"))
    memory.where(memory.table.body == "cat").update({"body": "kitten"})
    memory.where(memory.table.body == "dog").delete()
    script = (
        "from pixelmemory import Memory; from pixelmemory.context import Text; "
        f"m = Memory([Text(id='body', embed=False)], namespace={namespace!r}); "
        "assert [row['body'] for row in m.collect()] == ['kitten']"
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize("limit", [-1, 1.5, True], ids=["negative", "fraction", "bool"])
def test_invalid_search_limit_is_rejected(limit, namespace):
    memory = Memory([Text(id="body", embed=False)], namespace=namespace)
    with pytest.raises((TypeError, ValueError), match="limit"):
        memory.search("cat", limit=limit)


def test_zero_limit_returns_no_results(namespace):
    memory = Memory(
        [Text(id="body", embed_model=regression_embedding)], namespace=namespace
    )
    assert memory.search("cat", limit=0) == []
