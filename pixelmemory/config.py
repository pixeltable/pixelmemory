from typing import Literal, Optional
import pixeltable as pxt
from dataclasses import dataclass, field

SchemaType = Literal[
    pxt.Array,
    pxt.Audio,
    pxt.Bool,
    pxt.Date,
    pxt.Document,
    pxt.Float,
    pxt.Image,
    pxt.Int,
    pxt.Json,
    pxt.String,
    pxt.Timestamp,
    pxt.Video,
]


@dataclass
class AudioSplitterParams:
    duration: float = 30.0
    overlap: float = 0.0
    min_segment_duration: float = 0.0


@dataclass
class DocumentSplitterParams:
    separators: str = "token_limit"
    limit: Optional[int] = 300
    overlap: Optional[int] = None
    metadata: str = ""
    skip_tags: list[str] = field(default_factory=lambda: ["nav"])
    tiktoken_encoding: str = "cl100k_base"
    tiktoken_target_model: Optional[str] = None


@dataclass
class FrameIteratorParams:
    fps: Optional[float] = None
    num_frames: Optional[int] = None


@dataclass
class StringSplitterParams:
    separators: str = "sentence"


@dataclass
class WhisperParams:
    language: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None


@dataclass
class ChunkView:
    name: str
    table: pxt.Table


@dataclass
class FrameView:
    name: str
    table: pxt.Table


@dataclass(frozen=True)
class SearchTarget:
    """One searchable place: a column carrying an embedding index.

    A memory can have several. A chunked document puts its index on the chunk
    view's `text`; an image puts one on the generated description and optionally
    a CLIP index on the image itself; a video contributes both its transcript
    and its frame captions. `Memory.search()` walks these.

    `index_name` is recorded rather than inferred because one column can carry
    two indexes: chunked text is indexed on the chunk view AND directly on the
    base column, so a bare `.similarity()` there would be ambiguous.
    """

    context_id: str
    table: pxt.Table
    column: str
    index_name: str
