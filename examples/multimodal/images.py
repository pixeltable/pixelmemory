from pixelmemory import Memory
from pixelmemory.context import Image, Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Image(id="image", provider="openai", model="gpt-4o-mini"),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="image_memory_example",
    table_name="image_files"
)

image_url = "https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images/000000000030.jpg"

entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    image=image_url,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query = "A person on a skateboard"
sim = memory.image_description.similarity(query)
results = (
    memory.order_by(sim, asc=False)
    .limit(3)
    .select(memory.image, memory.image_description, similarity=sim)
    .collect()
)

for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Image: {res['image']}")
    print(f"Description: {res['image_description']}\n")
