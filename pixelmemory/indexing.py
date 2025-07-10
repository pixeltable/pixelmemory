import pixeltable as pxt
from typing import Any, Optional
import pixeltable as pxt
import dataclasses
from .memory import Memory
from .context import (
    Image,
    Audio,
    Video,
    String,
    Document,
)
from .config import (
    ChunkView,
    FrameView,
)
from .vision import (
    get_vision_function,
    prepare_vision_args,
    create_vision_computed_column,
)


def setup_column_indexing(
    memory_instance: Memory,
    col_name: str,
    col_type: Any,
    col_settings: Optional[Any] = None,
) -> None:
    embed_model = memory_instance._get_embed_model(col_settings.embedding_model)
    index_name = col_settings.index_name or "similarity"

    if col_type == pxt.Image:
        setup_image_indexing(
            memory_instance, col_name, embed_model, index_name, col_settings
        )
    elif col_type == pxt.Audio:
        setup_audio_indexing(
            memory_instance,
            col_name,
            embed_model,
            index_name,
            col_settings,
            audio_col=None,
        )
    elif col_type == pxt.Video:
        setup_video_indexing(
            memory_instance, col_name, embed_model, index_name, col_settings
        )
    elif col_type == pxt.String:
        setup_string_indexing(
            memory_instance, col_name, embed_model, index_name, col_settings
        )
    elif col_type == pxt.Document:
        setup_document_indexing(
            memory_instance, col_name, embed_model, index_name, col_settings
        )


def setup_vision_indexing(
    target_obj: pxt.Table,
    img_col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: Image,
) -> None:
    vision_func = get_vision_function(col_settings.provider)
    vision_args = prepare_vision_args(
        col_settings.provider,
        col_settings.model,
        col_settings.prompt,
        col_settings.llm_kwargs,
        img_col_name,
        target_obj,
    )

    description_col_name = create_vision_computed_column(
        col_settings.provider, img_col_name, vision_func, vision_args, target_obj
    )

    target_obj.add_embedding_index(
        column=description_col_name,
        idx_name=index_name,
        embedding=embed_model,
        if_exists="ignore",
    )

    if col_settings.use_clip:
        from pixeltable.functions.huggingface import clip

        target_obj.add_embedding_index(
            column=img_col_name,
            idx_name=f"{index_name}_clip",
            embedding=clip.using(model_id=col_settings.clip_model),
            if_exists="ignore",
        )


def setup_document_indexing(
    memory_instance: Memory,
    col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: Document,
) -> None:
    from pixeltable.iterators import DocumentSplitter

    chunk_view_name = f"{memory_instance.table_name}_{col_name}_chunks"
    chunk_view_path = f"{memory_instance.namespace}.{chunk_view_name}"

    document_source = getattr(memory_instance.table, col_name)

    chunk_view = pxt.create_view(
        chunk_view_path,
        memory_instance.table,
        iterator=DocumentSplitter.create(
            document=document_source, **dataclasses.asdict(col_settings.chunk_params)
        ),
        if_exists="replace_force",
    )
    if not chunk_view:
        chunk_view = pxt.get_table(chunk_view_path)

    memory_instance.resources.chunk_views.append(
        ChunkView(name=col_name, table=chunk_view)
    )

    chunk_view.add_embedding_index(
        column="text", idx_name=index_name, embedding=embed_model, if_exists="ignore"
    )


def setup_image_indexing(
    memory_instance: Memory,
    col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: Image,
) -> None:
    setup_vision_indexing(
        memory_instance.table,
        col_name,
        embed_model,
        index_name,
        col_settings,
    )


def setup_audio_indexing(
    memory_instance: Memory,
    col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: Audio,
    audio_col: Optional[pxt.Column] = None,
) -> None:
    from pixeltable.iterators import AudioSplitter, StringSplitter
    from pixeltable.functions.openai import transcriptions

    transcription_kwargs = {
        k: v
        for k, v in dataclasses.asdict(col_settings.transcription_kwargs).items()
        if v is not None
    }

    audio_chunk_view_name = f"{memory_instance.table_name}_{col_name}_audio_chunks"
    audio_chunk_view_path = f"{memory_instance.namespace}.{audio_chunk_view_name}"

    audio_source = (
        audio_col if audio_col is not None else getattr(memory_instance.table, col_name)
    )

    audio_chunk_view = pxt.create_view(
        audio_chunk_view_path,
        memory_instance.table,
        iterator=AudioSplitter.create(
            audio=audio_source, **dataclasses.asdict(col_settings.chunk_params)
        ),
        if_exists="replace_force",
    )
    if not audio_chunk_view:
        audio_chunk_view = pxt.get_table(audio_chunk_view_path)

    transcription_col_name = f"{col_name}_transcription"
    whisper_args = {
        "audio": audio_chunk_view.audio_chunk,
        "model": col_settings.transcription_model,
        **transcription_kwargs,
    }

    audio_chunk_view.add_computed_column(
        **{transcription_col_name: transcriptions(**whisper_args)}, if_exists="ignore"
    )

    sentence_view_name = f"{memory_instance.table_name}_{col_name}_sentence_chunks"
    sentence_view_path = f"{memory_instance.namespace}.{sentence_view_name}"

    transcription_text_col = getattr(audio_chunk_view, transcription_col_name).text

    sentence_chunk_view = pxt.create_view(
        sentence_view_path,
        audio_chunk_view,
        iterator=StringSplitter.create(
            text=transcription_text_col, separators="sentence"
        ),
        if_exists="replace_force",
    )
    if not sentence_chunk_view:
        sentence_chunk_view = pxt.get_table(sentence_view_path)

    memory_instance.resources.chunk_views.append(
        ChunkView(name=col_name, table=sentence_chunk_view)
    )

    sentence_chunk_view.add_embedding_index(
        column="text", idx_name=index_name, embedding=embed_model, if_exists="ignore"
    )


def setup_video_indexing(
    memory_instance: Memory,
    col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: Video,
) -> None:
    from pixeltable.functions.video import extract_audio
    from pixeltable.iterators import FrameIterator

    audio_col_name = f"{col_name}_audio"
    memory_instance.table.add_computed_column(
        **{audio_col_name: extract_audio(getattr(memory_instance.table, col_name))},
        if_exists="ignore",
    )
    audio_col = getattr(memory_instance.table, audio_col_name)
    audio_col_settings = Audio(
        name=col_name,
        chunk_params=col_settings.audio_chunk_params,
        transcription_model=col_settings.transcription_model,
        transcription_kwargs=col_settings.transcription_kwargs,
    )
    setup_audio_indexing(
        memory_instance,
        col_name,
        embed_model,
        index_name,
        audio_col_settings,
        audio_col=audio_col,
    )

    frame_view_name = f"{memory_instance.table_name}_{col_name}_frames"
    frame_view_path = f"{memory_instance.namespace}.{frame_view_name}"

    frame_view = pxt.create_view(
        frame_view_path,
        memory_instance.table,
        iterator=FrameIterator.create(
            video=getattr(memory_instance.table, col_name),
            **dataclasses.asdict(col_settings.frame_params),
        ),
        if_exists="ignore",
    )
    if not frame_view:
        frame_view = pxt.get_table(frame_view_path)
    memory_instance.resources.frame_views.append(
        FrameView(name=col_name, table=frame_view)
    )

    image_col_settings = Image(
        name="frame",
        provider=col_settings.provider,
        model=col_settings.model,
        prompt=col_settings.prompt,
        llm_kwargs=col_settings.llm_kwargs,
        use_clip=col_settings.use_clip,
        clip_model=col_settings.clip_model,
    )

    setup_vision_indexing(
        frame_view,
        "frame",
        embed_model,
        index_name,
        image_col_settings,
    )


def setup_string_indexing(
    memory_instance: Memory,
    col_name: str,
    embed_model: pxt.Function,
    index_name: str,
    col_settings: String,
) -> None:
    if col_settings.use_chunking:
        from pixeltable.iterators import StringSplitter

        chunk_view_name = f"{memory_instance.table_name}_{col_name}_chunks"
        chunk_view_path = f"{memory_instance.namespace}.{chunk_view_name}"

        text_source = getattr(memory_instance.table, col_name)

        chunk_view = pxt.create_view(
            chunk_view_path,
            memory_instance.table,
            iterator=StringSplitter.create(
                text=text_source, **dataclasses.asdict(col_settings.chunk_params)
            ),
            if_exists="replace_force",
        )
        if not chunk_view:
            chunk_view = pxt.get_table(chunk_view_path)

        memory_instance.resources.chunk_views.append(
            ChunkView(name=col_name, table=chunk_view)
        )

        chunk_view.add_embedding_index(
            column="text",
            idx_name=index_name,
            embedding=embed_model,
            if_exists="ignore",
        )

        memory_instance.table.add_embedding_index(
            column=col_name,
            idx_name=f"{index_name}_direct",
            embedding=embed_model,
            if_exists="ignore",
        )
    else:
        memory_instance.table.add_embedding_index(
            column=col_name,
            idx_name=index_name,
            embedding=embed_model,
            if_exists="replace_force",
        )
