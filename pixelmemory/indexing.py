import logging
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from .vision import get_vision_function, prepare_vision_args, create_vision_computed_column

logger = logging.getLogger(__name__)

def initialize_automatic_indexing(memory_instance) -> None:
    if not memory_instance.columns_to_index:
        return
    
    memory_instance.logger.info("Initializing automatic indexing.")
    
    if isinstance(memory_instance.text_embedding_model, str):
        embed_model_instance = sentence_transformer.using(model_id=memory_instance.text_embedding_model)
    else:
        embed_model_instance = memory_instance.text_embedding_model

    for col_name in memory_instance.columns_to_index:
        if col_name in memory_instance.schema:
            col_type = memory_instance.schema[col_name]
            memory_instance.logger.info(f"Setting up indexing for column '{col_name}' of type {col_type}.")
            
            if col_type == pxt.Image:
                setup_image_indexing(memory_instance, col_name, embed_model_instance)
            
            elif col_type == pxt.Audio:
                setup_audio_indexing(memory_instance, col_name, embed_model_instance)

            elif col_type == pxt.Video:
                setup_video_indexing(memory_instance, col_name, embed_model_instance)
            
            elif col_type == pxt.String:
                if memory_instance.use_sentence_chunking:
                    setup_string_sentence_chunking(memory_instance, col_name, embed_model_instance)
                else:
                    memory_instance.logger.info(f"Adding string embedding index to '{col_name}'.")
                    memory_instance.table.add_embedding_index(
                        column=col_name, 
                        idx_name=memory_instance.idx_name, 
                        embedding=embed_model_instance, 
                        if_exists="replace_force"
                    )
            
            elif col_type == pxt.Document:
                setup_document_indexing(memory_instance, col_name, embed_model_instance)

    memory_instance.logger.info("Automatic indexing initialization complete.")

def setup_document_indexing(memory_instance, col_name: str, embed_model_instance) -> None:
    memory_instance.logger.info(f"Setting up document indexing for column '{col_name}'.")
    from pixeltable.iterators import DocumentSplitter

    chunk_view_name = f"{memory_instance.table_name}_{col_name}_chunks"
    chunk_view_path = f"{memory_instance.namespace}.{chunk_view_name}"
    
    document_source = getattr(memory_instance.table, col_name)

    memory_instance.logger.info(f"Creating document chunk view: {chunk_view_path}")
    
    if not memory_instance.document_iterator_kwargs:
        memory_instance.document_iterator_kwargs = {'separators': 'token_limit', 'limit': 300}
        
    chunk_view = pxt.create_view(
        chunk_view_path,
        memory_instance.table,
        iterator=DocumentSplitter.create(document=document_source, **memory_instance.document_iterator_kwargs),
        if_exists="replace_force"
    )
    
    memory_instance.chunk_views[col_name] = chunk_view

    memory_instance.logger.info("Adding embedding index to 'text' column of document chunk view.")
    chunk_view.add_embedding_index(
        column="text",
        idx_name=memory_instance.idx_name,
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    memory_instance.logger.info(f"Document indexing setup for '{col_name}' complete.")

def setup_image_indexing(memory_instance, col_name: str, embed_model_instance) -> None:
    memory_instance.logger.info(f"Setting up image indexing for column '{col_name}'.")
    
    vision_func = get_vision_function(memory_instance)
    vision_args = prepare_vision_args(memory_instance, col_name, memory_instance.table)
    
    description_col_name = create_vision_computed_column(
        memory_instance, col_name, vision_func, vision_args, memory_instance.table
    )
    memory_instance.logger.info(f"Created vision description column: '{description_col_name}'.")
    
    memory_instance.logger.info(f"Adding embedding index to '{description_col_name}'.")
    memory_instance.table.add_embedding_index(
        column=description_col_name,
        idx_name=memory_instance.idx_name,
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    
    if memory_instance.use_clip:
        from pixeltable.functions.huggingface import clip
        memory_instance.logger.info(f"Adding CLIP embedding index to '{col_name}'.")
        memory_instance.table.add_embedding_index(
            column=col_name,
            idx_name=f"{memory_instance.idx_name}_clip",
            embedding=clip.using(model_id=memory_instance.clip_model),
            if_exists="ignore"
        )
    memory_instance.logger.info(f"Image indexing setup for '{col_name}' complete.")

def setup_audio_indexing(memory_instance, col_name: str, embed_model_instance, audio_col=None) -> None:
    memory_instance.logger.info(f"Setting up audio indexing for column '{col_name}'.")
    from pixeltable.iterators import AudioSplitter, StringSplitter
    from pixeltable.functions.openai import transcriptions

    audio_chunk_view_name = f"{memory_instance.table_name}_{col_name}_audio_chunks"
    audio_chunk_view_path = f"{memory_instance.namespace}.{audio_chunk_view_name}"
    
    audio_source = audio_col if audio_col is not None else getattr(memory_instance.table, col_name)

    memory_instance.logger.info(f"Creating audio chunk view: {audio_chunk_view_path}")
    audio_chunk_view = pxt.create_view(
        audio_chunk_view_path,
        memory_instance.table,
        iterator=AudioSplitter.create(audio=audio_source, **memory_instance.audio_iterator_kwargs),
        if_exists="replace_force"
    )
    
    transcription_col_name = f"{col_name}_transcription"
    whisper_args = {
        "audio": audio_chunk_view.audio_chunk, 
        "model": memory_instance.transcription_model, 
        **memory_instance.transcription_kwargs
    }
    
    memory_instance.logger.info(f"Adding transcription column '{transcription_col_name}' to audio chunk view.")
    audio_chunk_view.add_computed_column(
        **{transcription_col_name: transcriptions(**whisper_args)},
        if_exists="ignore"
    )
    
    sentence_view_name = f"{memory_instance.table_name}_{col_name}_sentence_chunks"
    sentence_view_path = f"{memory_instance.namespace}.{sentence_view_name}"
    
    transcription_text_col = getattr(audio_chunk_view, transcription_col_name).text
    
    memory_instance.logger.info(f"Creating sentence chunk view: {sentence_view_path}")
    sentence_chunk_view = pxt.create_view(
        sentence_view_path,
        audio_chunk_view,
        iterator=StringSplitter.create(text=transcription_text_col, separators="sentence"),
        if_exists="replace_force"
    )
    
    memory_instance.chunk_views[col_name] = sentence_chunk_view

    memory_instance.logger.info("Adding embedding index to 'text' column of sentence chunk view.")
    sentence_chunk_view.add_embedding_index(
        column="text",
        idx_name=memory_instance.idx_name,
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    memory_instance.logger.info(f"Audio indexing setup for '{col_name}' complete.")

def setup_video_indexing(memory_instance, col_name: str, embed_model_instance) -> None:
    memory_instance.logger.info(f"Setting up video indexing for column '{col_name}'.")
    from pixeltable.functions.video import extract_audio
    from pixeltable.iterators import FrameIterator

    audio_col_name = f"{col_name}_audio"
    memory_instance.logger.info(f"Extracting audio to column '{audio_col_name}'.")
    memory_instance.table.add_computed_column(
        **{audio_col_name: extract_audio(getattr(memory_instance.table, col_name))},
        if_exists="ignore"
    )
    setup_audio_indexing(memory_instance, col_name, embed_model_instance, audio_col=getattr(memory_instance.table, audio_col_name))

    frame_view_name = f"{memory_instance.table_name}_{col_name}_frames"
    frame_view_path = f"{memory_instance.namespace}.{frame_view_name}"
    
    memory_instance.logger.info(f"Creating frame view: {frame_view_path}")
    frame_view = pxt.create_view(
        frame_view_path,
        memory_instance.table,
        iterator=FrameIterator.create(video=getattr(memory_instance.table, col_name), **memory_instance.video_iterator_kwargs),
        if_exists="ignore"
    )
    memory_instance.frame_views[col_name] = frame_view
    
    vision_func = get_vision_function(memory_instance)
    vision_args = prepare_vision_args(memory_instance, 'frame', frame_view)
    description_col_name = create_vision_computed_column(
        memory_instance, 'frame', vision_func, vision_args, frame_view
    )
    memory_instance.logger.info(f"Created frame description column: '{description_col_name}'.")
    
    memory_instance.logger.info(f"Adding embedding index to '{description_col_name}'.")
    frame_view.add_embedding_index(
        column=description_col_name,
        idx_name=memory_instance.idx_name,
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    
    if memory_instance.use_clip:
        from pixeltable.functions.huggingface import clip
        memory_instance.logger.info("Adding CLIP embedding index to 'frame' column of frame view.")
        frame_view.add_embedding_index(
            column='frame',
            idx_name=f"{memory_instance.idx_name}_clip",
            embedding=clip.using(model_id=memory_instance.clip_model),
            if_exists="ignore"
        )
    memory_instance.logger.info(f"Video indexing setup for '{col_name}' complete.")

def setup_string_sentence_chunking(memory_instance, col_name: str, embed_model_instance) -> None:
    memory_instance.logger.info(f"Setting up sentence chunking for string column '{col_name}'.")
    from pixeltable.iterators import StringSplitter

    sentence_view_name = f"{memory_instance.table_name}_{col_name}_sentence_chunks"
    sentence_view_path = f"{memory_instance.namespace}.{sentence_view_name}"
    
    text_source = getattr(memory_instance.table, col_name)
    
    memory_instance.logger.info(f"Creating sentence chunk view: {sentence_view_path}")
    sentence_chunk_view = pxt.create_view(
        sentence_view_path,
        memory_instance.table,
        iterator=StringSplitter.create(text=text_source),
        if_exists="replace_force"
    )
    
    # Store the chunk view for later access
    memory_instance.chunk_views[col_name] = sentence_chunk_view

    memory_instance.logger.info("Adding embedding index to 'text' column of sentence chunk view.")
    sentence_chunk_view.add_embedding_index(
        column="text",
        idx_name=memory_instance.idx_name,
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    
    # Also add a direct embedding index to the original string column for flexibility
    memory_instance.logger.info(f"Adding direct embedding index to original string column '{col_name}'.")
    memory_instance.table.add_embedding_index(
        column=col_name,
        idx_name=f"{memory_instance.idx_name}_direct",
        embedding=embed_model_instance,
        if_exists="ignore"
    )
    
    memory_instance.logger.info(f"String sentence chunking setup for '{col_name}' complete.")
