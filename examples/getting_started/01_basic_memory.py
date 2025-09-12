from pixelmemory import Memory
from pixelmemory.context import Text

# Simple text memory without embedding (to avoid environment issues)
context = [
    Text(id="caption", embed=False),  # No embedding to avoid spacy issues
    Text(id="category", embed=False),
]

mem = Memory(context=context, namespace="basic_example")

entry_1 = mem.Entry(caption="This is a test.", category="test")
entry_2 = mem.Entry(caption="This is another test.", category="test")
entry_3 = mem.Entry(caption="Learning about memory systems.", category="learning")

mem.add(entry_1, entry_2, entry_3)

# Basic retrieval and filtering
all_results = mem.collect()
test_results = mem.where(mem.category == "test").collect()

print(f"Total entries: {len(all_results)}")
print(f"Test entries: {len(test_results)}")
for result in test_results:
    print(f"Caption: {result['caption']}, Category: {result['category']}")