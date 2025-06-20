import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime

website_schema = {
    'memory_id': pxt.Required[pxt.String],
    'website_content': pxt.Document,
    'inserted_at': pxt.Timestamp,
}

memory = Memory(
    namespace='website_memory_example',
    table_name='website_files',
    schema=website_schema,
    columns_to_index=['website_content'],
    primary_key='memory_id'
)

website_url = "https://quotes.toscrape.com/"
memory.insert([{
    'memory_id': str(uuid.uuid4()),
    'website_content': website_url,
    'inserted_at': datetime.now()
}])

query = "inspirational quotes about life"
chunk_view = memory.chunk_views['website_content']
sim = chunk_view.text.similarity(query)
results = chunk_view.order_by(sim, asc=False).limit(3).select(chunk_view.text, similarity=sim).collect()

for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text']}\n")
