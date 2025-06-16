from typing import Dict, Literal, Optional, Any, List
import uuid
import datetime

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
        if_exists: Literal["error", "ignore", "replace_force"] = "error",
    ):
        self.namespace = namespace
        self.table_name = table_name
        self.schema = schema
        self.if_exists = if_exists

        default_schema = {
            "memory_id": pxt.Required[pxt.String],
            "insert_at": pxt.Required[pxt.Timestamp],
            "content": pxt.Required[pxt.String],
            "metadata": pxt.Json,
        }

        pxt.create_dir(self.namespace, if_exists=self.if_exists)

        t = pxt.create_table(
            f"{self.namespace}.{self.table_name}",
            schema=default_schema,
            primary_key="memory_id",
            if_exists=self.if_exists,
        )

        if self.schema:
            t.add_columns(self.schema, if_exists=self.if_exists)

        self.table = t

    def add_entry(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        memory_id = uuid.uuid4().hex
        data = {
            "memory_id": memory_id,
            "insert_at": datetime.datetime.now(),
            "content": content,
        }
        if metadata:
            data.update(metadata)
        self.table.insert([data])
        return memory_id

    def add_entries(self, entries: List[Dict[str, Any]]) -> List[str]:
        memory_ids = []
        records = []
        for entry in entries:
            memory_id = uuid.uuid4().hex
            memory_ids.append(memory_id)
            record = {
                'memory_id': memory_id,
                'insert_at': datetime.datetime.now(),
                'content': entry['content'],
            }
            if 'metadata' in entry and entry.get('metadata'):
                record.update(entry['metadata'])
            records.append(record)

        if records:
            self.table.insert(records)

        return memory_ids