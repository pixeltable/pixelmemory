# Pixelmemory: Your Local-First AI Memory Layer

**Simple, powerful, and local-first memory for your LLM agents, powered by Pixeltable.**

Pixelmemory provides a Pythonic and intuitive way to add long-term memory capabilities to your Large Language Model (LLM) applications and agents. It's designed for developers who prioritize data ownership, control, and the flexibility to leverage the full power of their underlying data store. By building on [Pixeltable](https://pixeltable.readme.io/), Pixelmemory offers robust storage and efficient querying, all within your local environment.

## Core Philosophy

- **Developer Control**: You (or your agent) decide how memories are created, enriched, and linked. Pixelmemory provides the tools.
- **Local-First**: Your data stays with you. Pixelmemory operates on your local Pixeltable instance.
- **Simplicity for Core Tasks**: A clean API for the fundamental operation of adding memories.
- **Power Through Pixeltable**: Direct access to the underlying Pixeltable table for advanced querying and data manipulation.

## Key Features

- **Intuitive Python API**: Add memories with simple `add_entry()` and `add_entries()` methods.
- **Flexible Metadata**: Define a custom schema for metadata to store structured data alongside each memory.
- **Batch Operations**: Efficiently add multiple memories at once.
- **Namespaced Memory**: Maintain separate memory spaces using a `namespace` and `table_name` identifier (e.g., per user, per agent, per project).
- **Direct Pixeltable Access**: The `Memory` object exposes the underlying `Pixeltable` table, allowing you to use its full feature set for complex queries, updates, and data analysis.

## Installation

```bash
pip install pixelmemory
# Ensure you have Pixeltable installed and configured as per its documentation.
```

## Usage Example

Here is a trivial example of how to use `pixelmemory`.

```python
from pixelmemory import Memory
import pixeltable as pxt

# 1. Initialize the memory store
# This creates a pixeltable table named 'my_agent_memory' in the 'agent_1' namespace.
# It also adds a custom metadata column 'source' of type string.
print("Initializing Memory...")
memory = Memory(
    namespace="agent_1",
    table_name="my_agent_memory",
    metadata={"source": pxt.String},
    if_exists="replace_force"  # Use 'replace_force' for this example to ensure it runs cleanly
)

# 2. Add a single memory entry
print("\nAdding a single entry...")
memory.add_entry(
    content="The user's favorite color is blue.",
    metadata={"source": "conversation_1"}
)

# 3. Add multiple entries in a batch
print("\nAdding multiple entries...")
entries_to_add = [
    {
        "content": "The user is interested in machine learning.",
        "metadata": {"source": "conversation_2"}
    },
    {
        "content": "The user asked for a summary of the latest news.",
        "metadata": {"source": "conversation_3"}
    }
]
memory.add_entries(entries_to_add)

# 4. Access the underlying Pixeltable table to view and query memories
print("\nAll memories stored:")
print(memory.table.collect())

# You can use the full power of Pixeltable for advanced querying
print("\nQuerying for memories from 'conversation_1':")
query_result = memory.table.where(memory.table.source == 'conversation_1').collect()
print(query_result)
