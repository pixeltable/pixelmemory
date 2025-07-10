from typing import Dict, Any, Literal, List, Union, Callable, Optional
from dataclasses import dataclass
import pixeltable as pxt
from .config import (
    SchemaType,
    Text,
    Vision,
    Audio,
    Video,
    Document,
    ColumnsToEmbed,
    ChunkView,
    FrameView,
    IndexedColumn,
)


@dataclass
class MemoryResources:
    main_table: pxt.Table
    chunk_views: List[ChunkView]
    frame_views: List[FrameView]
    indexed_columns: List[IndexedColumn]


class Memory:
    def __init__(
        self,
        schema: Dict[str, SchemaType],
        namespace: str = "default_memory",
        table_name: str = "memory",
        columns_to_embed: ColumnsToEmbed = ColumnsToEmbed(),
        text: Text = Text(),
        vision: Vision = Vision(),
        audio: Audio = Audio(),
        video: Video = Video(),
        document: Document = Document(),
        if_exists: Literal["ignore", "error", "replace_force"] = "ignore",
        **kwargs,
    ):
        self.namespace = namespace
        self.table_name = table_name
        self.schema = schema
        self.columns_to_embed = columns_to_embed.columns
        self.text = text
        self.vision = vision
        self.audio = audio
        self.video = video
        self.document = document
        self.if_exists = if_exists

        table_path = f"{self.namespace}.{self.table_name}"
        if self.namespace not in pxt.list_dirs():
            pxt.create_dir(self.namespace)

        self.table: pxt.Table = pxt.create_table(
            table_path, schema=self.schema, if_exists=self.if_exists, **kwargs
        )

        self.resources = MemoryResources(
            main_table=self.table, chunk_views=[], frame_views=[], indexed_columns=[]
        )

        if self.columns_to_embed:
            self.setup_indexing()

    def _get_embed_model(
        self, override_model: Optional[Union[str, pxt.Function]] = None
    ) -> pxt.Function:
        model = override_model or self.text.embedding_model
        if isinstance(model, str):
            from pixeltable.functions.huggingface import sentence_transformer

            return sentence_transformer.using(model_id=model)
        # We have to assume it's a pxt.Function if it's not a string
        return model

    def setup_indexing(self, columns_to_index: Optional[List[str]] = None) -> None:
        from .indexing import setup_column_indexing

        columns_to_index = columns_to_index or list(self.columns_to_embed.keys())
        if not columns_to_index:
            return
        for col_name in columns_to_index:
            if col_name not in self.schema:
                continue
            col_type = self.schema[col_name]
            col_settings = self.columns_to_embed.get(col_name)
            setup_column_indexing(self, col_name, col_type, col_settings)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.resources.main_table, name):
            return getattr(self.resources.main_table, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object (or its underlying Table) has no attribute '{name}'"
        )
