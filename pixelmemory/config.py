from typing import Dict, Any, Literal, Union, Callable, Optional
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
class Audio:
    transcription_model: str = "whisper-1"
    transcription_kwargs: WhisperParams = field(default_factory=WhisperParams)
    chunk_params: AudioSplitterParams = field(
        default_factory=lambda: AudioSplitterParams(chunk_duration_sec=30.0)
    )


@dataclass
class Document:
    chunk_params: DocumentSplitterParams = field(
        default_factory=lambda: DocumentSplitterParams(limit=300)
    )


@dataclass
class Text:
    embedding_model: Union[str, Callable] = "intfloat/e5-large-v2"
    index_name: str = "similarity"
    use_chunking: bool = False
    chunk_params: StringSplitterParams = field(default_factory=StringSplitterParams)


@dataclass
class Video:
    frame_params: FrameIteratorParams = field(default_factory=FrameIteratorParams)


@dataclass
class Vision:
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    prompt: str = "Describe this image in detail, including colors, objects, scene, and any text visible."
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"


@dataclass
class Column:
    embedding_model: Union[str, Callable] = None
    index_name: str = None


@dataclass
class AudioColumn(Column):
    transcription_model: str = None
    transcription_kwargs: WhisperParams = None
    chunk_params: AudioSplitterParams = None


@dataclass
class DocumentColumn(Column):
    chunk_params: DocumentSplitterParams = None


@dataclass
class ImageColumn(Column):
    provider: Literal["openai", "anthropic"] = None
    model: str = None
    prompt: str = None
    llm_kwargs: Dict[str, Any] = None
    use_clip: bool = None
    clip_model: str = None


@dataclass
class StringColumn(Column):
    use_chunking: bool = None
    chunk_params: StringSplitterParams = None


@dataclass
class VideoColumn(Column):
    frame_params: FrameIteratorParams = None
    transcription_model: str = None
    transcription_kwargs: WhisperParams = None
    audio_chunk_params: AudioSplitterParams = None
    provider: Literal["openai", "anthropic"] = None
    model: str = None
    prompt: str = None
    llm_kwargs: Dict[str, Any] = None
    use_clip: bool = None
    clip_model: str = None


class ColumnsToEmbed:
    def __init__(self, **columns: Column):
        self.columns = columns


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
