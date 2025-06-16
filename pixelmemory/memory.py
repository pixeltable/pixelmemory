from typing import Dict, Literal, Optional

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
        namespace: str = 'default_memory',
        table_name: str = 'memory',
        schema: Optional[Dict[str, SchemaType]] = None,
        if_exists: Literal[
            'error', 'ignore', 'replace_force'
        ] = 'error'
    ):
        self.namespace = namespace
        self.table_name = table_name
        self.schema = schema
        self.if_exists = if_exists

        default_schema = {
            'memory_id': pxt.Required[pxt.String],
            'insert_at': pxt.Required[pxt.Timestamp],
            'content': pxt.Required[pxt.String],
            'metadata': pxt.Json,
        }

        if self.schema:
            default_schema.update(self.schema)

        pxt.create_dir(self.namespace, if_exists=self.if_exists)

        t = pxt.create_table(
            f'{self.namespace}.{self.table_name}',
            schema=default_schema,
            primary_key='memory_id',
            if_exists=self.if_exists,
        )

        self.table = t