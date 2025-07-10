import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime

memory = Memory(
    namespace='audio_memory_example',
    table_name='audio_files',
    schema={
        'memory_id': pxt.Required[pxt.String],
        'audio': pxt.Audio,
        'inserted_at': pxt.Timestamp,
    },
    columns_to_index=['audio'],
    primary_key='memory_id'
)

audio_url = "https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/10-minute%20tour%20of%20Pixeltable.mp3"
memory.insert([{
    'memory_id': str(uuid.uuid4()),
    'audio': audio_url,
    'inserted_at': datetime.now()
}])

query = "What are the key features of Pixeltable?"
chunk_view = memory.chunk_views['audio']
sim = chunk_view.text.similarity(query)
results = chunk_view.order_by(sim, asc=False).limit(3).select(chunk_view.text, similarity=sim).collect()

for res in results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text']}\n")
