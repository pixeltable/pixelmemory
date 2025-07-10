import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime

doc_schema = {
    "memory_id": pxt.Required[pxt.String],
    "document": pxt.Document,
    "inserted_at": pxt.Timestamp,
}

memory = Memory(
    namespace="document_memory_example",
    table_name="doc_files",
    schema=doc_schema,
    columns_to_index=["document"],
    primary_key="memory_id",
)

doc_url = "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/Zacks-Nvidia-Report.pdf"
memory.insert(
    [
        {
            "memory_id": str(uuid.uuid4()),
            "document": doc_url,
            "inserted_at": datetime.now(),
        }
    ]
)

query = "What are the key growth drivers for Nvidia?"
chunk_view = memory.chunk_views["document"]
sim = chunk_view.text.similarity(query)
results = (
    chunk_view.order_by(sim, asc=False)
    .limit(3)
    .select(chunk_view.text, similarity=sim)
    .collect()
)

for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text']}\n")
