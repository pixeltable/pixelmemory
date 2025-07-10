from pixelmemory import Memory
import pixeltable as pxt

schema = {"text": pxt.String, "image": pxt.Image}

mem = Memory(
    namespace="user_98283",
    table_name="memory",
    idx_name="conversation_embedding",
    schema=schema,
    columns_to_index=["text"],
)

mem.insert([{"text": "Hello, world!"}])

similarity = mem.text.similarity("Hello, world!")
result = (
    mem.order_by(similarity, asc=False)
    .select(mem.text, similarity=similarity)
    .collect()
)
print(result)
