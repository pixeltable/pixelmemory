from pixelmemory import Memory
from pixelmemory.context import Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Text(id="text_content", embed=True),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="text_memory_example",
    table_name="text_files"
)

text_data = (
    "Pixeltable is a powerful tool for multimodal data processing. "
    "It allows you to work with images, videos, audio, and documents seamlessly. "
    "Key features include automatic metadata extraction, computed columns, and vector search. "
    "You can build complex AI workflows with just a few lines of Python code."
)
entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    text_content=text_data,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query = "What can you do with Pixeltable?"

results = memory.search(query, on="text_content", limit=3, min_score=0.1)

print(f"Query: '{query}'")
print("Search results:")
for res in results:
    print(f"Similarity: {res['score']:.4f}")
    print(f"Text: {res['text'][:100]}...")
    print(f"ID: {res['memory_id']}")
    print("---")
