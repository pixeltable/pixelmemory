from pixelmemory import Memory
from pixelmemory.context import String, Image

context = [
    String(id="caption"),
    Image(id="image"),
    Video(id="youtube_video"),
]

mem = Memory(context=context)

print(
    f"Successfully created memory table with automatic text indexing at: {mem.table.memory_id}"
)

mem.insert(
    [
        {
            "caption": "This is a test.",
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        }
    ]
)

print(mem.select(mem.caption, mem.image, mem.youtube_video).collect())

mem.drop()
