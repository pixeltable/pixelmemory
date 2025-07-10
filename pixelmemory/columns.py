from typing import Dict, Any, Literal, Union, Optional
import pixeltable as pxt
from dataclasses import dataclass, field
from .config import (
    AudioSplitterParams,
    DocumentSplitterParams,
    FrameIteratorParams,
    StringSplitterParams,
    WhisperParams,
)


@dataclass
class Column:
    name: str
    embed: bool = True
    embedding_model: Optional[Union[str, pxt.Function]] = None
    index_name: Optional[str] = None


@dataclass
class Audio(Column):
    transcription_model: str = "whisper-1"
    transcription_kwargs: WhisperParams = field(default_factory=WhisperParams)
    chunk_params: AudioSplitterParams = field(
        default_factory=lambda: AudioSplitterParams(chunk_duration_sec=30.0)
    )
    _pxt_type: pxt.Audio = pxt.Audio


@dataclass
class Document(Column):
    chunk_params: DocumentSplitterParams = field(
        default_factory=lambda: DocumentSplitterParams(limit=300)
    )
    _pxt_type: pxt.Document = pxt.Document


@dataclass
class Image(Column):
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    prompt: str = "Describe this image in detail, including colors, objects, scene, and any text visible."
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    _pxt_type: pxt.Image = pxt.Image


@dataclass
class String(Column):
    use_chunking: bool = False
    chunk_params: StringSplitterParams = field(default_factory=StringSplitterParams)
    _pxt_type: pxt.String = pxt.String


@dataclass
class Video(Column):
    frame_params: FrameIteratorParams = field(default_factory=FrameIteratorParams)
    transcription_model: str = "whisper-1"
    transcription_kwargs: WhisperParams = field(default_factory=WhisperParams)
    audio_chunk_params: AudioSplitterParams = field(
        default_factory=lambda: AudioSplitterParams(chunk_duration_sec=30.0)
    )
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    prompt: str = "Describe this image in detail, including colors, objects, scene, and any text visible."
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    _pxt_type: pxt.Video = pxt.Video
