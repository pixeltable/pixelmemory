import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime

image_schema = {
    "memory_id": pxt.Required[pxt.String],
    "image": pxt.Image,
    "inserted_at": pxt.Timestamp,
}

memory = Memory(
    namespace="image_memory_example",
    table_name="image_files",
    schema=image_schema,
    columns_to_index=["image"],
    primary_key="memory_id",
)

image_url = "https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images/000000000030.jpg"
memory.insert(
    [
        {
            "memory_id": str(uuid.uuid4()),
            "image": image_url,
            "inserted_at": datetime.now(),
        }
    ]
)

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
