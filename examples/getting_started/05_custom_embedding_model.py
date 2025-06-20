from pixelmemory import Memory
import pixeltable as pxt

schema = {"text": pxt.String}

mem = Memory(
    schema=schema,
    columns_to_index=["text"],
    text_embedding_model="all-MiniLM-L6-v2",
    if_exists="replace_force"
)

mem.insert([{"text": "Hello, world!"}])

similarity = mem.text.similarity("Hello, world!")
result = (
    mem.order_by(similarity, asc=False)
    .select(mem.text, similarity=similarity)
    .collect()
)
print(result)