from typing import Dict, Any, Literal, List, Union, Optional, TYPE_CHECKING
from dataclasses import dataclass, make_dataclass, asdict
import pixeltable as pxt
from .config import (
    ChunkView,
    FrameView,
    SearchTarget,
)
from .context import Context

if TYPE_CHECKING:
    from dataclasses import dataclass as _dataclass_base
else:
    _dataclass_base = object


@dataclass
class MemoryResources:
    main_table: pxt.Table
    chunk_views: List[ChunkView]
    frame_views: List[FrameView]
    search_targets: List[SearchTarget]


class Memory:
    def __init__(
        self,
        context: List[Context],
        namespace: str = "default_memory",
        table_name: str = "memory",
        if_exists: Literal["ignore", "error", "replace_force"] = "ignore",
        **kwargs,
    ):
        self.namespace = namespace
        self.table_name = table_name
        self.context = context
        self.if_exists = if_exists

        self.schema: Dict[str, pxt.ColumnType] = {
            col.id: (col._pxt_type if col.required else col._pxt_type | None)
            for col in self.context
        }
        self.columns_to_embed: Dict[str, Context] = {
            col.id: col for col in self.context if col.embed
        }

        table_path = f"{self.namespace}.{self.table_name}"
        pxt.create_dir(self.namespace, if_exists="ignore", parents=True)

        self.table: pxt.Table = pxt.create_table(
            table_path, schema=self.schema, if_exists=self.if_exists, **kwargs
        )

        self.resources = MemoryResources(
            main_table=self.table, chunk_views=[], frame_views=[], search_targets=[]
        )

        if self.columns_to_embed:
            self.setup_indexing()

        self.Entry = make_dataclass(
            "MemoryEntry",
            [(col.id, Any) for col in self.context],
            bases=(_dataclass_base,),
        )

    def _get_embed_model(
        self, override_model: Optional[Union[str, pxt.Function]] = None
    ) -> pxt.Function:
        model = override_model or "intfloat/e5-large-v2"
        if isinstance(model, str):
            from pixeltable.functions.huggingface import sentence_transformer

            return sentence_transformer.using(model_id=model)
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

    def add(self, *rows: "Memory.Entry") -> None:
        """
        Add one or more rows to the memory table.

        This method supports batch insertion for efficiency, converting the provided
        Entry instances to the dictionary format required by Pixeltable.

        Args:
            *rows: One or more instances of the dynamically generated Memory.Entry dataclass.

        Raises:
            ValueError: If no rows are provided.

        Example:
            # Single row
            entry = self.Entry(caption="This is a test.", image_path="/path/to/image.jpg")
            self.add(entry)

            # Multiple rows (batched insert)
            entry1 = self.Entry(caption="Test 1", image_path="/path/1.jpg")
            entry2 = self.Entry(caption="Test 2", image_path="/path/2.jpg")
            self.add(entry1, entry2)
        """
        if not rows:
            raise ValueError("At least one row must be provided.")
        row_dicts = [asdict(row) for row in rows]
        self.table.insert(row_dicts)

    @property
    def search_targets(self) -> List[SearchTarget]:
        """Every indexed column this memory can be searched over."""
        return list(self.resources.search_targets)

    def views(self, context_id: Optional[str] = None) -> Dict[str, pxt.Table]:
        """
        The chunk and frame views backing this memory, by context id.

        Returns the raw Pixeltable handles, for queries `search()` does not
        cover. `memory.views()["report"]` is what the pre-0.2.0 examples were
        reaching for with `memory.chunk_views["report"]`, which never resolved:
        chunk_views is a list on `resources`, not a mapping on `Memory`.
        """
        out: Dict[str, pxt.Table] = {}
        for cv in self.resources.chunk_views:
            out[cv.name] = cv.table
        for fv in self.resources.frame_views:
            out[f"{fv.name}_frames"] = fv.table
        if context_id is None:
            return out
        if context_id not in out:
            raise KeyError(
                f"No view for {context_id!r}. Available: {sorted(out)}"
            )
        return {context_id: out[context_id]}

    def search(
        self,
        query: str,
        *,
        on: Optional[str] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[dict]:
        """
        Semantic search across this memory's indexed columns.

        Args:
            query: the text to search for
            on: restrict to one context id (e.g. "report", or "clip_frames"
                for a video's frame captions). Default searches every target.
            limit: maximum rows returned overall
            min_score: drop results below this similarity

        Returns dicts of {context_id, text, score}, best first.

        Replaces the order_by/similarity/limit block that every caller used to
        write by hand. `idx` is always passed, because one column can carry two
        indexes and a bare `.similarity()` on it is ambiguous.
        """
        targets = self.resources.search_targets
        if on is not None:
            targets = [t for t in targets if t.context_id == on]
            if not targets:
                available = sorted({t.context_id for t in self.resources.search_targets})
                raise KeyError(f"No search target {on!r}. Available: {available}")
        if not targets:
            raise RuntimeError(
                "This memory has no embedding index, so there is nothing to "
                "search. Pass embed=True on at least one Context."
            )

        rows: List[dict] = []
        for t in targets:
            col = getattr(t.table, t.column)
            sim = col.similarity(string=query, idx=t.index_name)
            results = (
                t.table.order_by(sim, asc=False)
                .limit(limit)
                .select(text=col, score=sim)
                .collect()
            )
            for r in results:
                if min_score is not None and r["score"] < min_score:
                    continue
                rows.append(
                    {"context_id": t.context_id, "text": r["text"], "score": r["score"]}
                )

        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:limit]

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs whenever normal lookup fails, including while __init__ is
        # still running or after it raised. Reading self.resources through it would
        # recurse; bail out to a plain AttributeError instead.
        if name.startswith("_") or "resources" not in self.__dict__:
            raise AttributeError(name)
        if hasattr(self.resources.main_table, name):
            return getattr(self.resources.main_table, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object (or its underlying Table) has no attribute '{name}'"
        )
