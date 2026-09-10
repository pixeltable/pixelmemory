from pixelmemory import Memory
from pixelmemory.context import Document, Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Document(id="document"),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="document_memory_example",
    table_name="doc_files"
)

doc_url = "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/Zacks-Nvidia-Report.pdf"

entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    document=doc_url,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query = "What are the key growth drivers for Nvidia?"
results = memory.search(query, on="document", limit=3)

for res in results:
    print(f"Similarity: {res['score']:.4f}")
    print(f"Text: {res['text']}\n")
