from pixelmemory import Memory
from pixelmemory.context import Document, Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Document(id="website_content"),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="website_memory_example",
    table_name="website_files"
)

website_url = "https://quotes.toscrape.com/"

entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    website_content=website_url,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query = "inspirational quotes about life"
results = memory.search(query, on="website_content", limit=3)

for res in results:
    print(f"Similarity: {res['score']:.4f}")
    print(f"Text: {res['text']}\n")
