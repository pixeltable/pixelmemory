import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime

text_schema = {
    'memory_id': pxt.Required[pxt.String],
    'text_content': pxt.String,
    'inserted_at': pxt.Timestamp,
}

memory = Memory(
    namespace='text_memory_example',
    table_name='text_files',
    schema=text_schema,
    columns_to_index=['text_content'],
    use_sentence_chunking=True,
    primary_key='memory_id'
)

text_data = (
    "Pixeltable is a powerful tool for multimodal data processing. "
    "It allows you to work with images, videos, audio, and documents seamlessly. "
    "Key features include automatic metadata extraction, computed columns, and vector search. "
    "You can build complex AI workflows with just a few lines of Python code."
)
memory.insert([{
    'memory_id': str(uuid.uuid4()),
    'text_content': text_data,
    'inserted_at': datetime.now()
}])

query = "What can you do with Pixeltable?"
chunk_view = memory.chunk_views['text_content']
sim = chunk_view.text.similarity(query)
results = chunk_view.order_by(sim, asc=False).limit(2).select(chunk_view.text, similarity=sim).collect()

for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text']}\n")
