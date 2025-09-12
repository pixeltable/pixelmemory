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

# Direct similarity search on text_content column
sim = memory.text_content.similarity(query)
results = (
    memory.where(sim >= 0.1)
    .order_by(sim, asc=False)
    .limit(3)
    .select(memory.text_content, memory.memory_id, similarity=sim)
    .collect()
)

print(f"Query: '{query}'")
print("Search results:")
for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text_content'][:100]}...")
    print(f"ID: {res['memory_id']}")
    print("---")
