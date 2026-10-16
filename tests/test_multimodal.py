"""Execute the real multimodal pipelines without provider calls or model downloads.

Only provider clients and the spaCy model loader are replaced. Pixeltable still
executes its own media iterators, provider UDFs, response conversion, indexes,
dependent views, and similarity queries.
"""

import base64
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import av
import numpy as np
import pixeltable as pxt
from PIL import Image as PILImage
import pytest

from pixelmemory import Memory
from pixelmemory.config import (
    AudioSplitterParams,
    DocumentSplitterParams,
    FrameIteratorParams,
)
from pixelmemory.context import Audio, Document, Image, Text, Video


@pxt.udf
def local_embedding(text: str) -> pxt.Array[(3,), np.float32]:
    """A deterministic index that needs no downloaded embedding model."""
    return np.array(
        [1.0, float("cat" in text.lower()), float("dog" in text.lower())],
        dtype=np.float32,
    )


@pytest.fixture
def sentence_model(monkeypatch):
    import spacy
    import pixeltable.functions.string as string_functions

    model = spacy.blank("en")
    model.add_pipe("sentencizer")
    monkeypatch.setattr(string_functions, "get_spacy_model", lambda name: model)


@pytest.fixture
def provider_requests(monkeypatch):
    import pixeltable.functions.anthropic as anthropic_functions
    import pixeltable.functions.openai as openai_functions

    requests = []

    async def transcribe(**kwargs):
        assert Path(kwargs["file"]).stat().st_size > 0
        requests.append({"kind": "transcription", **kwargs})
        return SimpleNamespace(
            model_dump=lambda **_: {"text": "A cat walks. A dog sleeps."}
        )

    async def openai_caption(**kwargs):
        image_url = kwargs["messages"][0]["content"][1]["image_url"]["url"]
        prefix, encoded = image_url.split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded).startswith(b"\x89PNG")
        requests.append({"kind": "openai", **kwargs})
        return SimpleNamespace(
            headers={},
            text=json.dumps(
                {"choices": [{"message": {"content": "A cat on a red mat."}}]}
            ),
        )

    async def anthropic_caption(**kwargs):
        source = kwargs["messages"][0]["content"][1]["source"]
        assert source["media_type"] == "image/png"
        assert base64.b64decode(source["data"]).startswith(b"\x89PNG")
        requests.append({"kind": "anthropic", **kwargs})

        async def text():
            return json.dumps(
                {"content": [{"type": "text", "text": "A cat on a red mat."}]}
            )

        return SimpleNamespace(headers={}, text=text)

    openai_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe)),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=openai_caption)
            )
        ),
    )
    anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=anthropic_caption)
        )
    )
    monkeypatch.setattr(openai_functions, "_openai_client", lambda: openai_client)
    monkeypatch.setattr(
        anthropic_functions, "_anthropic_client", lambda: anthropic_client
    )
    return requests


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.zeros(48000, dtype=np.int16).tobytes())
    return str(path)


def _video_file(path: Path, *, with_audio: bool) -> str:
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=2)
        video.width = 32
        video.height = 32
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("aac", rate=16000) if with_audio else None
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 30, dtype=np.uint8), format="rgb24"
            )
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if audio is not None:
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 48000), dtype=np.float32), format="fltp", layout="mono"
            )
            frame.sample_rate = 16000
            frame.pts = 0
            frame.time_base = Fraction(1, 16000)
            for packet in audio.encode(frame):
                output.mux(packet)
            for packet in audio.encode():
                output.mux(packet)
    return str(path)


def test_chunked_text_insert_search_and_delete(namespace, sentence_model):
    memory = Memory(
        [Text(id="body", use_chunking=True, embed_model=local_embedding)],
        namespace=namespace,
    )
    memory.add(memory.Entry(body="A cat walks. A dog sleeps."))
    chunks = memory.views()["body"]
    assert chunks.select(chunks.text).collect()["text"] == [
        "A cat walks.",
        "A dog sleeps.",
    ]
    hits = memory.search("cat", on="body", limit=1)
    assert hits[0]["text"] == "A cat walks."
    memory.table.delete()
    assert chunks.count() == 0
    assert memory.search("cat") == []


@pytest.mark.parametrize("extension", ["txt", "html", "pdf"])
def test_document_insert_search(namespace, tmp_path, extension):
    path = tmp_path / f"sample.{extension}"
    if extension == "pdf":
        # A tiny PDF fixture avoids adding a PDF authoring dependency to tests.
        stream = b"BT /F1 12 Tf 72 720 Td (A cat walks. A dog sleeps.) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        ]
        pdf = b"%PDF-1.4\n"
        offsets = []
        for number, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 6\n0000000000 65535 f \n"
        pdf += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
        pdf += (
            f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        path.write_bytes(pdf)
    elif extension == "html":
        path.write_text(
            "<html><nav>Navigation</nav><p>A cat walks.</p><p>A dog sleeps.</p></html>"
        )
    else:
        path.write_text("A cat walks. A dog sleeps.")
    memory = Memory(
        [
            Document(
                id="report",
                embed_model=local_embedding,
                chunk_params=DocumentSplitterParams(separators="char_limit", limit=15),
            )
        ],
        namespace=namespace,
    )
    memory.add(memory.Entry(report=str(path)))
    hits = memory.search("cat", on="report", limit=10)
    assert hits and any("cat" in hit["text"] for hit in hits)
    assert all("Navigation" not in hit["text"] for hit in hits)


def test_default_document_token_chunking(namespace, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("A cat walks. A dog sleeps. " * 200)
    memory = Memory(
        [Document(id="report", embed_model=local_embedding)], namespace=namespace
    )
    memory.add(memory.Entry(report=str(path)))
    chunks = memory.views()["report"]
    assert chunks.count() > 1
    assert memory.search("cat", on="report")


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        pytest.param(Image, {}, id="image"),
        pytest.param(
            Image,
            {"provider": "anthropic", "model": "claude-sonnet-4-5"},
            id="anthropic",
        ),
        pytest.param(Audio, {}, id="audio"),
        pytest.param(Video, {}, id="video"),
        pytest.param(Document, {}, id="document"),
    ],
)
def test_nullable_media_skips_providers_and_keeps_targets(
    namespace, kind, kwargs, sentence_model, provider_requests
):
    memory = Memory(
        [kind(id="source", required=False, embed_model=local_embedding, **kwargs)],
        namespace=namespace,
    )
    assert memory.search_targets
    memory.add(memory.Entry())
    assert memory.table.count() == 1
    assert provider_requests == []
    assert memory.search("cat") == []
    assert all(view.count() == 0 for view in memory.views().values())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_image_caption_insert_and_reopen(namespace, provider, provider_requests):
    context = [
        Image(
            id="photo",
            embed_model=local_embedding,
            provider=provider,
            model="gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-5",
            llm_kwargs={"temperature": 0.2},
        )
    ]
    memory = Memory(context, namespace=namespace)
    memory.add(memory.Entry(photo=PILImage.new("RGB", (16, 16), "red")))
    assert memory.search("cat")[0]["text"] == "A cat on a red mat."
    assert provider_requests[0]["temperature"] == 0.2
    if provider == "anthropic":
        assert provider_requests[0]["max_tokens"] == 1024
    version = memory.table.get_metadata()["version"]
    reopened = Memory(context, namespace=namespace)
    assert reopened.table.get_metadata()["version"] == version
    assert reopened.search("cat") == memory.search("cat")
    assert len(provider_requests) == 1


def test_audio_insert_transcribes_segments_and_searches_sentences(
    namespace, audio_file, sentence_model, provider_requests
):
    memory = Memory(
        [
            Audio(
                id="recording",
                embed_model=local_embedding,
                chunk_params=AudioSplitterParams(duration=1.0),
            )
        ],
        namespace=namespace,
    )
    memory.add(memory.Entry(recording=audio_file))
    assert len(provider_requests) >= 2
    assert all(request["kind"] == "transcription" for request in provider_requests)
    hits = memory.search("cat", on="recording")
    assert hits and all(hit["text"] == "A cat walks." for hit in hits[:2])
    memory.table.delete()
    assert memory.views()["recording"].count() == 0


@pytest.mark.parametrize("with_audio", [False, True])
def test_video_insert_searches_frames_and_optional_audio(
    namespace, tmp_path, with_audio, sentence_model, provider_requests
):
    path = _video_file(tmp_path / "sample.mp4", with_audio=with_audio)
    memory = Memory(
        [
            Video(
                id="clip",
                embed_model=local_embedding,
                frame_params=FrameIteratorParams(num_frames=2),
            )
        ],
        namespace=namespace,
    )
    memory.add(memory.Entry(clip=path))
    assert memory.views()["clip_frames"].count() == 2
    assert memory.search("cat", on="clip_frames")[0]["text"] == "A cat on a red mat."
    transcripts = memory.search("cat", on="clip")
    assert bool(transcripts) == with_audio
    assert sum(r["kind"] == "transcription" for r in provider_requests) == int(
        with_audio
    )
    memory.table.delete()
    assert all(view.count() == 0 for view in memory.views().values())


def test_provider_failure_aborts_insert(namespace, monkeypatch):
    import pixeltable.functions.openai as openai_functions

    def unavailable():
        raise RuntimeError("caption service unavailable")

    monkeypatch.setattr(openai_functions, "_openai_client", unavailable)
    memory = Memory(
        [Image(id="photo", embed_model=local_embedding)], namespace=namespace
    )
    with pytest.raises(pxt.Error, match="caption service unavailable"):
        memory.add(memory.Entry(photo=PILImage.new("RGB", (16, 16), "red")))
    assert memory.table.count() == 0
