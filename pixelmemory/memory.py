from typing import Dict, Literal, Optional, Any, List
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

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
    def __init__(self,
        namespace: str = "default_memory",
        table_name: str = "memory",
        schema: Optional[Dict[str, SchemaType]] = None, 
        columns_to_index: Optional[List[str]] = None,
        idx_name: Optional[str] = "similarity",
        embedding_model: Optional[str] = "intfloat/e5-large-v2",
        if_exists: Literal["ignore", "error", "replace_force"] = "ignore",
        **kwargs
    ):
        self.namespace: str = namespace
        self.table_name: str = table_name
        self.columns_to_index: Optional[List[str]] = columns_to_index
        self.idx_name: Optional[str] = idx_name
        self.embedding_model: Optional[str] = embedding_model
        
        table_path = f"{self.namespace}.{self.table_name}"
        if self.namespace not in pxt.list_dirs():
            pxt.create_dir(self.namespace)
        
        self.schema: Dict[str, SchemaType] = schema
        self.table: pxt.Table = pxt.create_table(
            table_path,
            schema=self.schema,
            if_exists=if_exists,
            **kwargs 
        )
        
        if self.columns_to_index:
            self._initialize_automatic_indexing()

    def _initialize_automatic_indexing(self) -> None:
        if not self.columns_to_index:
            return
        
        embed_model_instance = sentence_transformer.using(model_id=self.embedding_model)
        for col_name in self.columns_to_index:
            if col_name in self.schema: 
                self.table.add_embedding_index(column=col_name, idx_name=self.idx_name, string_embed=embed_model_instance, if_exists="ignore")

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.table, name):
            return getattr(self.table, name)
        raise AttributeError(f"'{self.__class__.__name__}' object (or its underlying Table) has no attribute '{name}'")
