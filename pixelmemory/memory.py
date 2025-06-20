import logging
from typing import Dict, Any, Literal, Optional, List, Union, Callable
import pixeltable as pxt


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

class Memory:
    def __init__(
        self,
        namespace: str = "default_memory",
        table_name: str = "memory",
        schema: Optional[Dict[str, SchemaType]] = None,
        columns_to_index: Optional[List[str]] = None,
        idx_name: Optional[str] = "similarity",
        text_embedding_model: Optional[Union[str, Callable]] = "intfloat/e5-large-v2",
        vision_provider: Literal["openai", "anthropic", "gemini"] = "openai",
        vision_model: str = "gpt-4o-mini",
        vision_prompt: str = "Describe this image in detail, including colors, objects, scene, and any text visible.",
        vision_kwargs: Optional[Dict[str, Any]] = None,
        use_clip: bool = False,
        clip_model: str = "openai/clip-vit-base-patch32",
        transcription_model: str = "whisper-1",
        transcription_kwargs: Optional[Dict[str, Any]] = None,
        audio_iterator_kwargs: Optional[Dict[str, Any]] = None,
        video_iterator_kwargs: Optional[Dict[str, Any]] = None,
        document_iterator_kwargs: Optional[Dict[str, Any]] = None,
        use_sentence_chunking: bool = False,
        if_exists: Literal["ignore", "error", "replace_force"] = "ignore",
        log_level: Union[str, int] = logging.WARNING,
        **kwargs,
    ):
        self.namespace = namespace
        self.table_name = table_name
        self.schema = schema
        self.columns_to_index = columns_to_index
        self.idx_name = idx_name
        self.text_embedding_model = text_embedding_model
        self.vision_provider = vision_provider
        self.vision_model = vision_model
        self.vision_prompt = vision_prompt
        self.vision_kwargs = vision_kwargs if vision_kwargs is not None else {}
        self.use_clip = use_clip
        self.clip_model = clip_model
        self.transcription_model = transcription_model
        self.transcription_kwargs = transcription_kwargs if transcription_kwargs is not None else {}
        self.audio_iterator_kwargs = audio_iterator_kwargs if audio_iterator_kwargs is not None else {'chunk_duration_sec': 30.0}
        self.video_iterator_kwargs = video_iterator_kwargs if video_iterator_kwargs is not None else {}
        self.document_iterator_kwargs = document_iterator_kwargs if document_iterator_kwargs is not None else {}
        self.use_sentence_chunking = use_sentence_chunking
        self.if_exists = if_exists
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if self.schema is None:
            self.logger.info("No schema provided, using default schema.")
            self.schema = {
                'memory_id': pxt.Required[pxt.String],
                'content': pxt.String,
                'insert_at': pxt.Timestamp,
            }
            if self.columns_to_index is None:
                self.columns_to_index = ['content']
            
            if 'primary_key' not in kwargs:
                kwargs['primary_key'] = 'memory_id'

        self.logger.info(f"Initializing Memory with namespace='{self.namespace}' and table_name='{self.table_name}'")

        self.chunk_views: Dict[str, pxt.View] = {}
        self.frame_views: Dict[str, pxt.View] = {}
        
        table_path = f"{self.namespace}.{self.table_name}"
        if self.namespace not in pxt.list_dirs():
            self.logger.info(f"Creating directory: {self.namespace}")
            pxt.create_dir(self.namespace)
        
        self.logger.info(f"Ensuring table '{table_path}' exists with if_exists='{self.if_exists}'.")

        self.table: pxt.Table = pxt.create_table(
            table_path,
            schema=self.schema,
            if_exists=self.if_exists,
            **kwargs
        )

        if self.columns_to_index:
            from .indexing import initialize_automatic_indexing
            self.logger.info("Proceeding with automatic indexing.")
            initialize_automatic_indexing(self)
            
        self.logger.info("Memory initialization complete.")

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.table, name):
            return getattr(self.table, name)
        raise AttributeError(f"'{self.__class__.__name__}' object (or its underlying Table) has no attribute '{name}'")
