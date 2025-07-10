import pixeltable as pxt
from pixelmemory import Memory
import uuid
from datetime import datetime
from pixeltable.functions.openai import embeddings

video_schema = {
    "memory_id": pxt.Required[pxt.String],
    "video": pxt.Video,
    "inserted_at": pxt.Timestamp,
}

memory = Memory(
    namespace="video_memory_example",
    table_name="video_files",
    schema=video_schema,
    columns_to_index=["video"],
    text_embedding_model=embeddings.using(model="text-embedding-3-small"),
    primary_key="memory_id",
)

video_url = "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/Lex-Fridman-Podcast-430-Excerpt-0.mp4"
memory.insert(
    [
        {
            "memory_id": str(uuid.uuid4()),
            "video": video_url,
            "inserted_at": datetime.now(),
        }
    ]
)

query_audio = "What is the guest's perspective on AI?"
query_visual = "A person gesturing with their hands"

audio_chunk_view = memory.chunk_views["video"]
audio_sim = audio_chunk_view.text.similarity(query_audio)
audio_results = (
    audio_chunk_view.order_by(audio_sim, asc=False)
    .limit(2)
    .select(audio_chunk_view.text, similarity=audio_sim)
    .collect()
)

frame_view = memory.frame_views["video"]
frame_sim = frame_view.frame_description.similarity(query_visual)
visual_results = (
    frame_view.order_by(frame_sim, asc=False)
    .limit(2)
    .select(frame_view.frame, frame_view.frame_description, similarity=frame_sim)
    .collect()
)

print(f"\nAudio search results for: '{query_audio}'\n")
for res in audio_results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text: {res['text']}\n")

print(f"\nVisual search results for: '{query_visual}'\n")
for res in visual_results:
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Frame: {res['frame']}")
    print(f"Description: {res['frame_description']}\n")
