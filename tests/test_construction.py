"""Construction smoke tests: no API key, no insert, no network call to a provider.

Every test here builds a Memory and inspects the catalog metadata Pixeltable
recorded. That is enough to catch the whole class of breakage this suite exists
for: a renamed iterator kwarg, a view rebuilt on every construction, a column
that should accept nulls and does not.
"""

import os

import pytest

import pixeltable as pxt

from pixelmemory import Memory
from pixelmemory.config import WhisperParams
from pixelmemory.context import Audio, Document, Image, Text, Video


def _make(kind, embed_model, **kw):
    return kind(id=kind.__name__.lower(), embed_model=embed_model, **kw)


def test_no_api_keys_in_env() -> None:
    """The suite's own precondition, asserted rather than assumed."""
    assert "OPENAI_API_KEY" not in os.environ
    assert "ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("kind", [Text, Document, Audio, Video, Image])
def test_every_context_type_constructs(kind, embed_model, namespace) -> None:
    """Document, Audio and Video all raised on Pixeltable 0.7.6 before this commit."""
    memory = Memory(context=[_make(kind, embed_model)], namespace=namespace)
    assert memory.table.get_metadata()["path"] == f"{namespace}/memory"


def test_audio_iterator_uses_current_kwarg_names(embed_model, namespace) -> None:
    """`iterator_call` is the printed form of what the catalog stored.

    It names the kwargs verbatim, so it is the most precise available check that
    `chunk_duration_sec`/`overlap_sec`/`min_chunk_duration_sec` really became
    `duration`/`overlap`/`min_segment_duration`.
    """
    Memory(context=[Audio(id="aud", embed_model=embed_model)], namespace=namespace)
    call = pxt.get_table(f"{namespace}.memory_aud_audio_chunks").get_metadata()[
        "iterator_call"
    ]
    assert call.startswith("audio_splitter(")
    assert "duration=30.0" in call
    assert "min_segment_duration=" in call
    assert "overlap=" in call
    for gone in ("chunk_duration_sec", "overlap_sec", "min_chunk_duration_sec"):
        assert gone not in call


def test_document_iterator_uses_current_kwarg_names(embed_model, namespace) -> None:
    Memory(context=[Document(id="doc", embed_model=embed_model)], namespace=namespace)
    call = pxt.get_table(f"{namespace}.memory_doc_chunks").get_metadata()[
        "iterator_call"
    ]
    assert call.startswith("document_splitter(")
    assert "skip_tags=['nav']" in call
    assert "html_skip_tags" not in call


def test_audio_view_exposes_audio_segment(embed_model, namespace) -> None:
    """The AudioSplitter output column was `audio_chunk`; it is now `audio_segment`."""
    Memory(context=[Audio(id="aud", embed_model=embed_model)], namespace=namespace)
    columns = pxt.get_table(f"{namespace}.memory_aud_audio_chunks").get_metadata()[
        "columns"
    ]
    assert "audio_segment" in columns
    assert "audio_chunk" not in columns


def test_reconstruction_does_not_rebuild_views(embed_model, namespace) -> None:
    """`if_exists='replace_force'` gave every view a new id on every construction,
    which meant re-chunking and re-embedding everything. Ids must be stable."""
    context = [
        Text(id="txt", embed_model=embed_model, use_chunking=True),
        Audio(id="aud", embed_model=embed_model),
    ]
    Memory(context=context, namespace=namespace)
    paths = [
        f"{namespace}.memory",
        f"{namespace}.memory_txt_chunks",
        f"{namespace}.memory_aud_audio_chunks",
        f"{namespace}.memory_aud_sentence_chunks",
    ]
    before = {p: pxt.get_table(p).get_metadata()["id"] for p in paths}

    Memory(context=context, namespace=namespace)
    after = {p: pxt.get_table(p).get_metadata()["id"] for p in paths}

    assert before == after


def test_reconstruction_does_not_rebuild_embedding_index(
    embed_model, namespace
) -> None:
    """The unchunked-text path used `replace_force` on `add_embedding_index`, so a
    second construction re-embedded the whole table."""
    context = [Text(id="txt", embed_model=embed_model)]
    Memory(context=context, namespace=namespace)
    table = pxt.get_table(f"{namespace}.memory")
    version_before = table.get_metadata()["version"]

    Memory(context=context, namespace=namespace)
    assert pxt.get_table(f"{namespace}.memory").get_metadata()["version"] == (
        version_before
    )


def test_openai_vision_uses_chat_completions(embed_model, namespace) -> None:
    """`pixeltable.functions.openai.vision` is deprecated. The replacement takes a
    messages list with an image_url block, and `llm_kwargs` go to `model_kwargs`."""
    memory = Memory(
        context=[
            Image(id="img", embed_model=embed_model, llm_kwargs={"temperature": 0.2})
        ],
        namespace=namespace,
    )
    columns = memory.table.get_metadata()["columns"]
    call = columns["img_response"]["computed_with"]
    assert call.startswith("chat_completions(")
    assert "'type': 'image_url'" in call
    assert "b64_encode(img, 'png')" in call
    assert "model_kwargs={'temperature': 0.2}" in call
    assert (
        columns["img_description"]["computed_with"]
        == "img_response.choices[0].message.content.astype(String)"
    )


def test_anthropic_vision_passes_max_tokens(embed_model, namespace) -> None:
    """`anthropic.messages` requires `max_tokens`, and the image must be base64
    text rather than a raw Image ColumnRef."""
    memory = Memory(
        context=[
            Image(
                id="img",
                provider="anthropic",
                model="claude-sonnet-4-5",
                embed_model=embed_model,
                llm_kwargs={"temperature": 0.2},
            )
        ],
        namespace=namespace,
    )
    call = memory.table.get_metadata()["columns"]["img_response"]["computed_with"]
    assert call.startswith("messages(")
    assert "max_tokens=1024" in call
    assert "'data': b64_encode(img, 'png')" in call
    assert "model_kwargs={'temperature': 0.2}" in call


def test_whisper_kwargs_go_to_model_kwargs(embed_model, namespace) -> None:
    """`transcriptions` is `(audio, model, model_kwargs)`. Splatting WhisperParams
    at the top level raised."""
    Memory(
        context=[
            Audio(
                id="aud",
                embed_model=embed_model,
                transcription_kwargs=WhisperParams(language="en"),
            )
        ],
        namespace=namespace,
    )
    columns = pxt.get_table(f"{namespace}.memory_aud_audio_chunks").get_metadata()[
        "columns"
    ]
    assert columns["aud_transcription"]["computed_with"] == (
        "transcriptions(audio=audio_segment, model='whisper-1', "
        "model_kwargs={'language': 'en'})"
    )


def test_nested_namespace(embed_model) -> None:
    """`pxt.list_dirs()` returns slash-separated paths, so the old
    `namespace not in pxt.list_dirs()` check never matched a dotted namespace and
    the second construction failed."""
    ns = "org.team.project"
    Memory(context=[Text(id="txt", embed_model=embed_model)], namespace=ns)
    Memory(context=[Text(id="txt", embed_model=embed_model)], namespace=ns)
    assert pxt.get_table(f"{ns}.memory").get_metadata()["path"] == (
        "org/team/project/memory"
    )


def test_optional_column_accepts_null(namespace) -> None:
    """Pixeltable columns are non-nullable by default since 0.7.3."""
    memory = Memory(
        context=[
            Text(id="body", embed=False),
            Text(id="note", embed=False, required=False),
        ],
        namespace=namespace,
    )
    columns = memory.table.get_metadata()["columns"]
    assert columns["body"]["type_"] == "String"
    assert columns["note"]["type_"] == "String | None"

    memory.table.insert([{"body": "hello", "note": None}])
    assert memory.table.count() == 1


def test_required_column_rejects_null(namespace) -> None:
    memory = Memory(
        context=[Text(id="body", embed=False)],
        namespace=namespace,
    )
    with pytest.raises(pxt.Error):
        memory.table.insert([{"body": None}])


def test_getattr_passthrough_still_works(embed_model, namespace) -> None:
    """`memory.where(...)` / `memory.collect()` are used throughout the README."""
    memory = Memory(context=[Text(id="txt", embed=False)], namespace=namespace)
    assert memory.collect() is not None
    with pytest.raises(AttributeError):
        getattr(memory, "definitely_not_a_table_method")


def test_failed_init_gives_attribute_error_not_recursion_error() -> None:
    """A Memory whose __init__ never set `resources` used to turn any attribute
    access into infinite recursion, hiding the real construction failure."""
    broken = Memory.__new__(Memory)
    with pytest.raises(AttributeError):
        getattr(broken, "collect")
    # Dunder lookups (copy, pickle, repr helpers) took the same recursive path.
    # hasattr only swallows AttributeError, so a RecursionError still fails here.
    assert not hasattr(broken, "__deepcopy__")
