# Pixelmemory: Your Local-First AI Memory Layer

**Simple, powerful, and local-first memory for your LLM agents, powered by Pixeltable.**

Pixelmemory provides a Pythonic and intuitive way to add long-term memory capabilities to your Large Language Model (LLM) applications and agents. It's designed for developers who prioritize data ownership, control, and the flexibility to leverage the full power of their underlying data store. By building on [Pixeltable](https://pixeltable.readme.io/), Pixelmemory offers robust storage and efficient semantic search, all within your local environment.

## Core Philosophy

- **Developer Control**: You (or your agent) decide how memories are created, enriched, and linked. Pixelmemory provides the tools.
- **Local-First**: Your data stays with you. Pixelmemory operates on your local Pixeltable instance.
- **Simplicity for Core Tasks**: A clean API for common memory operations.
- **Power Through Pixeltable**: Direct access to underlying Pixeltable tables for advanced querying.

## Key Features

- **Intuitive Python API**: Manage memories with simple method calls.
- **Semantic Search**: Find relevant text-based memories using vector similarity.
- **Optional Semantic Query**: Search by metadata and time filters even without a text query.
- **Unique Memory IDs**: Each memory entry is assigned a unique ID for precise operations.
- **Full CRUD Operations**: Create, Read, Update, and Delete individual memories.
- **Batch Operations**: Efficiently add or delete multiple memories at once.
- **Flexible Metadata**: Store arbitrary structured JSON data alongside each memory entry.
- **Namespaced Memory**: Maintain separate memory spaces using a `namespace` identifier (e.g., per user, per agent, per project).
- **Advanced Pixeltable Access**: An "escape hatch" to use the full power of Pixeltable.

## Installation

```bash
pip install pixelmemory
# Ensure you have Pixeltable installed and configured as per its documentation.
```

## Quick Start

```python
from pixelmemory import Memory
import datetime # For timestamp examples

# Initialize the Memory system
memory_store = Memory()
```

## Class Definition
```python
from typing import List, Dict, Any, Optional, Union # Added for clarity
import datetime

class Memory:
    def __init__(
        self, 
        name: str = "default_memory", 
        schema: Optional[Dict] = None,
        embedding_config: Optional[Union[str, Dict]] = "intfloat/e5-large-v2"
    ):
        """
        Initializes the Memory system.

        Args:
            name: The name of the memory store.
            schema: Optional schema for the memory store.
            embedding_config: Configuration for the embedding model.
                Can be a string (e.g., a Hugging Face model ID for sentence-transformers)
                or a dictionary for more detailed Pixeltable embedding function setup.
                Defaults to "intfloat/e5-large-v2".
        """
        # ... implementation details ...
        pass

    # --------------------
    # Adding Memories
    # --------------------
    def add_entry(self, content: str, namespace: str = "default_namespace", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Adds a single piece of text content as a memory entry.
        The 'content' is embedded for semantic search.
        A unique memory_id is generated and returned.

        Args:
            content: The text content to be stored and embedded.
            namespace: Identifier to partition memory spaces (e.g., per user, agent, project).
            metadata: Optional dictionary for storing additional structured data.

        Returns:
            The unique memory_id (str) for the newly added entry.
        """
        # ... implementation details ...
        pass

    def add_entries(self, entries: List[Dict[str, Any]], namespace: str = "default_namespace") -> List[str]:
        """
        Adds multiple memory entries in a batch.

        Args:
            entries: A list of dictionaries, where each dictionary must contain
                     'content' (str) and can optionally contain 'metadata' (dict).
            namespace: Identifier to partition memory spaces.

        Returns:
            A list of unique memory_ids (str) for the newly added entries.
        """
        # ... implementation details ...
        pass

    # --------------------
    # Retrieving Memories
    # --------------------
    def get_memory(self, memory_id: str, namespace: str = "default_namespace") -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific memory entry by its unique memory_id.

        Args:
            memory_id: The unique ID of the memory to retrieve.
            namespace: The namespace where the memory resides.

        Returns:
            A dictionary containing the memory data (timestamp, memory_id, content, metadata)
            or None if the memory_id is not found.
        """
        # ... implementation details ...
        pass

    def search(
        self,
        namespace: str = "default_namespace",
        query: Optional[str] = None,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for relevant memories.
        If 'query' is provided, semantic similarity search is performed.
        Filtering can also be done by metadata and/or timestamp range.
        Timestamp filters 'start_time' (greater than or equal to) and 
        'end_time' (less than or equal to) can be used individually 
        (for "since" or "before" a point in time) or together to define a specific range.

        Args:
            namespace: The namespace to search within.
            query: Optional search query text for semantic search.
            limit: Maximum number of results to return.
            metadata_filter: Optional dictionary to filter memories by exact matches
                             in their metadata.
            start_time: Optional datetime to retrieve memories created on or after this time.
            end_time: Optional datetime to retrieve memories created on or before this time.

        Returns:
            A list of result dictionaries, each containing:
            'memory_id', 'timestamp', 'content', 'metadata'.
            If 'query' was provided, 'similarity_score' is also included.
        """
        # ... implementation details ...
        pass

    # --------------------
    # Updating Memories
    # --------------------
    def update_memory(
        self,
        memory_id: str,
        namespace: str = "default_namespace",
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Updates the content and/or metadata of an existing memory entry.
        If 'content' is updated, it will be re-embedded.
        Only fields that are provided (not None) will be updated.

        Args:
            memory_id: The unique ID of the memory to update.
            namespace: The namespace where the memory resides.
            content: Optional new text content for the memory.
            metadata: Optional new metadata. This typically replaces existing metadata.

        Returns:
            True if the update was successful, False otherwise.
        """
        # ... implementation details ...
        pass

    # --------------------
    # Deleting Memories
    # --------------------
    def delete_memory(self, memory_id: str, namespace: str = "default_namespace") -> bool:
        """
        Deletes a specific memory entry by its unique memory_id.

        Args:
            memory_id: The unique ID of the memory to delete.
            namespace: The namespace where the memory resides.

        Returns:
            True if deletion was successful, False otherwise.
        """
        # ... implementation details ...
        pass

    def delete_memories(self, memory_ids: List[str], namespace: str = "default_namespace") -> int:
        """
        Deletes multiple memory entries by their unique memory_ids in a batch.

        Args:
            memory_ids: A list of unique memory IDs to delete.
            namespace: The namespace where the memories reside.

        Returns:
            The number of memories successfully deleted.
        """
        # ... implementation details ...
        pass

    # --------------------
    # Advanced Access
    # --------------------
    def get_namespace_table(self, namespace: str): # -> pxt.Table (actual type from Pixeltable)
        """
        Provides direct access to the underlying Pixeltable table for a given namespace.
        This is an "escape hatch" for advanced users who want to perform
        custom queries or leverage Pixeltable-specific features.

        Args:
            namespace: The identifier for the namespace whose table is to be accessed.

        Returns:
            The pixeltable.Table object for the namespace's memory data.
        """
        # ... implementation details that return the Pixeltable Table object ...
        pass
```
## Usage Examples (Pixelmemory API)

```python
from pixelmemory import Memory
import datetime

# Initialize
memory_store = Memory()
my_project_namespace = "project_alpha_research"

# 1. Add a research note
note_content = "Found a significant paper on topic Z by author X, published in 2023."
entry_id1 = memory_store.add_entry(
    content=note_content,
    namespace=my_project_namespace,
    metadata={"type": "literature_note", "author": "X", "year": 2023, "tags": ["topic_Z", "key_paper"]}
)
print(f"Added memory with ID: {entry_id1}")

# 2. Search for notes related to "topic Z" without a semantic query, only metadata
print(f"\nSearching for notes in '{my_project_namespace}' tagged 'topic_Z' (metadata only):")
results_meta_only = memory_store.search(
    namespace=my_project_namespace,
    metadata_filter={"tags": "topic_Z"} # Assuming metadata search can handle list containment or exact match
)
for res in results_meta_only:
    print(f"  ID: {res['memory_id']}, Content: {res['content'][:50]}...")


# 3. Search semantically for papers by "author X"
print(f"\nSearching for papers by 'author X' in '{my_project_namespace}':")
results_semantic = memory_store.search(
    namespace=my_project_namespace,
    query="papers by author X", # Semantic query
    limit=3
)
for res in results_semantic:
    print(f"  ID: {res['memory_id']}, Content: {res['content'][:50]}..., Score: {res['similarity_score']:.4f}")

# 4. Temporal Filtering: Retrieve all notes from the last 7 days
print(f"\nRetrieving all notes from '{my_project_namespace}' in the last 7 days:")
one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
recent_notes = memory_store.search(
    namespace=my_project_namespace,
    start_time=one_week_ago 
    # No query, no metadata_filter, just time-based
)
for res in recent_notes:
    print(f"  ID: {res['memory_id']}, Timestamp: {res['timestamp']}, Content: {res['content'][:30]}...")
```
## Handling Multi-Modal Data

Pixelmemory's core API is text-focused for direct embedding and semantic search. However, you can manage and associate multi-modal data (images, audio, video, documents) using Pixelmemory as the metadata and text-description store.

### 1. References and Textual Descriptions in Metadata

- Process your multi-modal file (e.g., an image or document).
- Generate a textual description or summary of the multi-modal content using an appropriate model (e.g., an image captioning model, document summarizer).
- Store this textual description in `pixelmemory` via `add_entry()`, making it semantically searchable.
- The `metadata` field of this memory entry will contain:
  - The file path, URL, or unique ID of the original multi-modal file.
  - Any extracted features or annotations from the multi-modal file.

**Example:**

```python
# An agent processes an image: /path/to/image_001.jpg
image_description = "A detailed schematic of a new circuit board design." # Generated by agent's vision/analysis model
image_memory_id = memory_store.add_entry(
    content=image_description, # This text is embedded and searched
    namespace="project_beta_designs",
    metadata={
        "type": "design_schematic_image",
        "original_file_path": "/path/to/image_001.jpg",
        "version": "v2.1",
        "components_highlighted": ["cpu_socket", "ram_slots"] 
    }
)
# When this memory is retrieved by search, the agent can use 
# metadata["original_file_path"] to access the actual image.
```

### 2. Leveraging Pixeltable's Multi-Modal Capabilities via `get_namespace_table()`

- If your Pixeltable instance is configured for direct multi-modal data storage, embedding, and search, you can:
- Use `memory_store.get_namespace_table(namespace=...)` to get direct access to the Pixeltable table.
- Interact with Pixeltable using its native API to insert multi-modal data, generate multi-modal embeddings, and perform multi-modal queries.
- This is an advanced pattern where an application or agent directly orchestrates Pixeltable's richer functionalities for specific multi-modal tasks, while still using Pixelmemory for managing associated textual descriptions or metadata.

## Contributing

(Details to be added)

## License

(Specify your license, e.g., Apache 2.0, MIT)
