from pixelmemory import Memory
from pixelmemory.context import Audio, Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Audio(id="audio", transcription_model="whisper-1"),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="audio_memory_example",
    table_name="audio_files"
)

audio_url = "https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/10-minute%20tour%20of%20Pixeltable.mp3"

entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    audio=audio_url,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query = "What are the key features of Pixeltable?"
chunk_view = memory.chunk_views["audio"]
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
