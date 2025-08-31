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
    chunk_duration_sec: float = 30.0
    overlap_sec: float = 0.0
    min_chunk_duration_sec: float = 0.0


@dataclass
class DocumentSplitterParams:
    separators: str = "token_limit"
    limit: Optional[int] = 300
    overlap: Optional[int] = None
    metadata: str = ""
    html_skip_tags: list[str] = field(default_factory=lambda: ["nav"])
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


@dataclass
class IndexedColumn:
    original_col: str
    indexed_col: str
