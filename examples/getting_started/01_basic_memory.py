from pixelmemory import Memory
from pixelmemory.context import Text, Video
from pixelmemory.config import StringSplitterParams

context = [
    Text(id="caption",  chunk_params=StringSplitterParams(limit=100)),
    Video(id="video", chunk_params=FrameIteratorParams(fps=1)),
]

mem = Memory(context=context)

entry_1 = mem.Entry(caption="This is a test.")
entry_2 = mem.Entry(caption="This is another test.")
entry_3 = mem.Entry(caption="s3://test.mp4")

mem.add(entry_1, entry_2, entry_3)

print(mem.select(mem.caption).collect())