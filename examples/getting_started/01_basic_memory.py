from pixelmemory import Memory
from pixelmemory.context import Text

context = [
    Text(id="caption", embed=True),
]

mem = Memory(context=context)

entry_1 = mem.Entry(caption="This is a test.")
entry_2 = mem.Entry(caption="This is another test.")
entry_3 = mem.Entry(caption="Learning about memory systems.")

mem.add(entry_1, entry_2, entry_3)

# Query with semantic search
similarity = mem.caption.similarity("testing")
results = (
    mem.where(similarity >= 0.1)
    .order_by(similarity, asc=False)
    .select(mem.caption, similarity=similarity)
    .collect()
)

print("Results with similarity scores:")
for result in results:
    print(f"Caption: {result['caption']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print("---")