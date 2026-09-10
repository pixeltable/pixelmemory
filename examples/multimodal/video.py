from pixelmemory import Memory
from pixelmemory.context import Video, Text
import uuid
from datetime import datetime

context = [
    Text(id="memory_id", embed=False),
    Video(id="video", provider="openai", model="gpt-4o-mini", embed_model="text-embedding-3-small"),
    Text(id="inserted_at", embed=False),
]

memory = Memory(
    context=context,
    namespace="video_memory_example",
    table_name="video_files"
)

video_url = "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/Lex-Fridman-Podcast-430-Excerpt-0.mp4"

entry = memory.Entry(
    memory_id=str(uuid.uuid4()),
    video=video_url,
    inserted_at=str(datetime.now())
)
memory.add(entry)

query_audio = "What is the guest's perspective on AI?"
query_visual = "A person gesturing with their hands"

audio_results = memory.search(query_audio, on="video", limit=2)

# Frame captions are registered separately, so visuals can be searched on their own.
visual_results = memory.search(query_visual, on="video_frames", limit=2)

print(f"\nAudio search results for: '{query_audio}'\n")
for res in audio_results:
    print(f"Similarity: {res['score']:.4f}")
    print(f"Text: {res['text']}\n")

print(f"\nVisual search results for: '{query_visual}'\n")
for res in visual_results:
    print(f"Similarity: {res['score']:.4f}")
    print(f"Description: {res['text']}\n")
# For the frame images themselves, take the raw view: memory.views("video_frames")
