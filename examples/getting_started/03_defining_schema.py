from pixelmemory import Memory
import pixeltable as pxt

schema = {
    "memory_id": pxt.String,
    "name": pxt.String,
    "content": pxt.String,
    "metadata": pxt.Json,
}

mem = Memory(schema=schema)

print(f"Successfully created memory table with custom schema at: {mem.memory_id}")

print(f"Table Information: {mem.table.get_metadata()}")
