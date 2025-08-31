<div align="center">
<img src="pixelmemory.png"
     alt="Pixelmemory Logo" width="70%" />
<br></br>

<h2>Open Source Multimodal Memory Layer for LLMs & Agents</h2>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pixelmemory?logo=python&logoColor=white&)
![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-E5DDD4)
<br>
[![PyPI Package](https://img.shields.io/pypi/v/pixelmemory?color=4D148C)](https://pypi.org/project/pixelmemory/)
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)


</div>

---

Pixelmemory is the only Python framework that provides persistent, searchable, and multimodal memory for your LLM agents and applications.

## 😩 Building AI Applications with Memory is Still Too Hard

Most AI applications today are essentially goldfish—they forget everything the moment a conversation ends:
*   Agents restart from scratch every time, losing valuable context and learned behaviors.
*   RAG systems can't handle complex multimodal data or maintain conversation history.
*   Expensive cloud memory solutions lock in your data with per-query fees and usage limits.
*   Complex infrastructure required to store, index, and retrieve memories across text, images, videos, and documents.
*   Limited semantic search capabilities for finding conceptually related memories.

This memory amnesia makes AI applications feel disconnected and limits their ability to build meaningful, persistent relationships with users.

## 💾 Installation

```python
pip install pixelmemory
```

**Pixelmemory is built on Pixeltable.** It stores memories and metadata persistently, typically in a `.pixeltable` directory in your workspace. Your data stays local and under your control.

## ⚙️ Audio Setup

After installation, you may need to download models for certain functionalities. For example, to use `spacy` for natural language processing, you need to download a language model:

```bash
uv run -- spacy download en_core_web_sm
```

This command uses `uv` to run the `spacy` download command within your project's virtual environment.

## ✨ What is Pixelmemory?

With Pixelmemory, you can give your LLMs and agents persistent, searchable memory that works across all data types. Built on **[Pixeltable](https://docs.pixeltable.com/)**, Pixelmemory automatically handles:

*   **Memory Storage & Retrieval:** Store conversations, interactions, and multimodal content with automatic timestamping.
*   **Semantic Search:** Find relevant memories by meaning, not just keywords, using built-in embedding indexes.
*   **Multimodal Support:** Handle text, images, videos, audio, and documents in a unified memory interface.
*   **Flexible Schemas:** Define custom memory structures that fit your exact application needs.
*   **Batch Operations:** Efficiently store and retrieve large volumes of memories.
*   **Temporal Queries:** Access memories from specific time periods or conversation contexts.
*   **Local-First:** Your data never leaves your infrastructure—no vendor lock-in or usage fees.

**Focus on your agent logic, not the memory infrastructure.**

## 🚀 Key Features

* **[Intuitive Python API:](https://docs.pixeltable.com/docs/datastore/tables-and-operations)** Simple `insert()` and `batch_insert()` methods for storing memories.
  ```python
  memory = Memory("chatbot", "conversations", schema)
  memory.insert({"role": "user", "content": "Remember this"})
  ```

* **[Semantic Memory Search:](https://docs.pixeltable.com/docs/datastore/embedding-index)** Built-in similarity search across your memory store.
  ```python
  # Find memories similar to current input
  similarity = memory.content.similarity("user's question")
  relevant_memories = memory.where(similarity >= 0.7).collect()
  ```

* **[Multimodal Memory Types:](https://docs.pixeltable.com/docs/datastore/bringing-data)** Store and search across all data modalities.
  ```python
  multimodal_schema = {
      'text': pxt.String,
      'image': pxt.Image,
      'video': pxt.Video,
      'audio': pxt.Audio
  }
  ```

* **[Flexible Memory Organization:](https://docs.pixeltable.com/docs/datastore/views)** Namespace and organize memories by agent, user, or conversation.
  ```python
  user_memory = Memory("agent_1", "user_123_history", schema)
  global_memory = Memory("agent_1", "knowledge_base", schema)
  ```

* **[Advanced Querying:](https://docs.pixeltable.com/docs/datastore/filtering-and-selecting)** Combine semantic search with filters and temporal queries.
  ```python
  recent_relevant = (
      memory
      .where((memory.timestamp > yesterday) & (similarity >= 0.8))
      .order_by(similarity, asc=False)
      .limit(5)
  )
  ```

* **[Direct Pixeltable Access:](https://docs.pixeltable.com/docs/datastore/custom-functions)** Full database power when you need advanced functionality.
  ```python
  # Memory object forwards to underlying Pixeltable table
  memory.show()  # Display memories
  memory.describe()  # Schema info
  memory.drop()  # Advanced operations
  ```

## Installation

```bash
pip install pixelmemory
```

## Examples

This project includes a variety of examples to help you get started and explore different features.

### Basic Usage (`examples/usage/basic/`)
- `insert.py`
- `search.py`

### Memory Types (`examples/memory_types/`)
- `agentic_memory.py`
- `long_term_conversational.py`
- `multimodal.py`
- `semantic_similiarity.py`
- `short_term_conversational.py`

### Multimodal (`examples/multimodal/`)
- `image.py`
- `text.py`
- `video.py`

### Integrations (`examples/integrations/`)
- `anthropic.py`
- `autogen.py`
- `groq.py`
- `oai.py`
- `pixelagent.py`

## Usage Example

This example demonstrates a simple chat application that uses Pixelmemory to store and retrieve conversation history for semantic search.

```python
from openai import OpenAI
import pixeltable as pxt
from pixelmemory import Memory
from datetime import datetime

openai_client = OpenAI()

chat_schema = {
    "user_id": pxt.String,
    "role": pxt.String,
    "content": pxt.String,
}

chat_memory = Memory(
    namespace="chatbot_basic",
    table_name="conversation_history",
    schema=chat_schema,
    columns_to_index=["content"],
    if_exists="ignore"
)

def chat_with_ai_and_memories(message: str, user_id: str = "default_user") -> str:
    similarity_score = chat_memory.content.similarity(message)
    
    relevant_memories_query = (
        chat_memory
        .where((chat_memory.user_id == user_id) & (similarity_score >= 0.7))
        .order_by(similarity_score, asc=False)
        .select(chat_memory.role, chat_memory.content)
        .limit(3)
    )
    
    retrieved_memories = relevant_memories_query.collect()

    memories_str = "\n".join(f"- {entry['role']}: {entry['content']}" for entry in retrieved_memories)
    if not memories_str:
        memories_str = "No relevant memories found."

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's question based on the current query and the following relevant past conversation snippets.\n"
        f"Relevant Past Conversation:\n{memories_str}"
    )
    
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=llm_messages
    )
    assistant_response = response.choices[0].message.content

    new_memories = [
        {"user_id": user_id, "role": "user", "content": message},
        {"user_id": user_id, "role": "assistant", "content": assistant_response}
    ]
    chat_memory.insert(new_memories)

    return assistant_response

def main():
    print("Chat with AI (type 'exit' to quit)")
    user_id_for_session = "test_user_123"
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        ai_reply = chat_with_ai_and_memories(user_input, user_id=user_id_for_session)
        print(f"AI: {ai_reply}")

if __name__ == "__main__":
    main()
```

## Advanced Search and Querying with Pixeltable

Pixelmemory allows direct access to the underlying Pixeltable API for powerful and flexible querying. This means you can leverage Pixeltable's full capabilities for semantic search, temporal filtering, and complex data retrieval.

Here's an example based on `examples/usage/basic/search.py`, demonstrating how to perform semantic and temporal searches:

```python
import uuid
from datetime import datetime, timedelta

import pixeltable as pxt
from pixelmemory import Memory

schema = {
    "memory_id": pxt.Required[pxt.String],
    "role": pxt.String,
    "content": pxt.String,
    "insert_at": pxt.Timestamp,
}

memory = Memory(
    namespace="chatbot_advanced_search",
    table_name="chat_history_advanced",
    schema=schema,
    columns_to_index=["content"],
    embedding_model="intfloat/e5-large-v2",
    if_exists="ignore",
)

memory_id_conv1 = uuid.uuid4().hex
memory_id_conv2 = uuid.uuid4().hex
memory_id_conv3 = uuid.uuid4().hex

data_to_insert = [
    {
        "memory_id": memory_id_conv1,
        "role": "user",
        "content": "Hello, how are you today?",
        "insert_at": datetime.now() - timedelta(minutes=10),
    },
    {
        "memory_id": memory_id_conv1,
        "role": "assistant",
        "content": "I am doing well! I learned that Pixeltable is a database for AI.",
        "insert_at": datetime.now() - timedelta(minutes=9),
    },
    {
        "memory_id": memory_id_conv2,
        "role": "user",
        "content": "What are the key features of Pixeltable?",
        "insert_at": datetime.now() - timedelta(days=2),
    },
    {
        "memory_id": memory_id_conv2,
        "role": "assistant",
        "content": "Pixeltable excels at handling multimodal data like images and videos, and supports time-travel queries.",
        "insert_at": datetime.now() - timedelta(days=2, minutes=-1),
    },
    {
        "memory_id": memory_id_conv3,
        "role": "user",
        "content": "Tell me about yesterday's discussion on project alpha.",
        "insert_at": datetime.now() - timedelta(hours=20),
    },
    {
        "memory_id": memory_id_conv3,
        "role": "assistant",
        "content": "Yesterday, we decided to use Python for project alpha's backend.",
        "insert_at": datetime.now() - timedelta(hours=19, minutes=50),
    }
]
memory.insert(data_to_insert)

query_text = "Information about Pixeltable features"
min_similarity_threshold = 0.7

similarity_score = memory.content.similarity(query_text)

relevant_memories = (
    memory.where(similarity_score >= min_similarity_threshold)
    .order_by(similarity_score, asc=False)
    .select(
        memory.role,
        memory.content,
        similarity=similarity_score
    )
    .limit(5)
)

print("Semantic Search Results:")
for row in relevant_memories.collect():
    print(f"  Role: {row['role']}, Content: '{row['content']}', Similarity: {row['similarity']:.4f}")

# Temporal Search: Find memories within a specific time window (e.g., last 12 to 36 hours)
start_time_window = datetime.now() - timedelta(hours=36)
end_time_window = datetime.now() - timedelta(hours=12)

memories_in_window = (
    memory.where((memory.insert_at >= start_time_window) & (memory.insert_at <= end_time_window))
    .order_by(memory.insert_at, asc=False)
    .select(
        memory.role,
        memory.content,
        memory.insert_at
    )
)

print(f"\nMemories from {start_time_window.strftime('%Y-%m-%d %H:%M')} to {end_time_window.strftime('%Y-%m-%d %H:%M')}:")
for row in memories_in_window.collect():
    print(f"  Role: {row['role']}, Content: '{row['content']}', Timestamp: {row['insert_at']}")

```
This demonstrates leveraging Pixeltable's native querying for advanced memory retrieval, providing fine-grained control over how memories are searched and filtered.

## 🖼️ Image Memory & Multimodal Search

Pixelmemory provides automatic image embedding and search capabilities using state-of-the-art vision models. When you include `pxt.Image` columns in your schema and `columns_to_index`, Pixelmemory automatically:

1. **Analyzes images** using vision models (OpenAI GPT-4o, Gemini, or Anthropic)
2. **Generates detailed descriptions** optimized for semantic search
3. **Creates embedding indexes** for fast similarity search
4. **Optionally adds CLIP embeddings** for direct image-to-image search

### Basic Image Memory Example

```python
import pixeltable as pxt
from pixelmemory import Memory

# Define schema with images
schema = {
    "image": pxt.Image,
    "caption": pxt.String,
    "tags": pxt.String,
}

# Create Memory with automatic image indexing
memory = Memory(
    namespace="visual_memory",
    table_name="photo_collection",
    schema=schema,
    columns_to_index=["image", "caption"],  # Index both images and text
    vision_provider="openai",  # or "gemini", "anthropic"
    vision_model="gpt-4o-mini",
    vision_prompt="Describe this image in detail, focusing on objects, colors, and scene.",
)

# Insert images - descriptions generated automatically
memory.insert([
    {
        "image": "path/to/garden.jpg",
        "caption": "Colorful flower garden", 
        "tags": "nature, flowers, outdoor"
    },
    {
        "image": "path/to/kitchen.jpg",
        "caption": "Modern kitchen interior",
        "tags": "interior, cooking, appliances"
    }
])

# Search images by content - finds relevant images even if caption doesn't match
query = "kitchen appliances and cooking"
similarity = memory.image_description.similarity(query)
results = (
    memory
    .where(similarity >= 0.3)
    .order_by(similarity, asc=False)
    .select(memory.caption, memory.tags, similarity=similarity)
    .collect()
)

for result in results:
    print(f"Found: {result['caption']} (similarity: {result['similarity']:.3f})")
```

### Advanced Image Features

```python
# Advanced configuration with custom vision parameters and CLIP
memory = Memory(
    namespace="advanced_visual",
    table_name="media_library",
    schema={"image": pxt.Image, "title": pxt.String},
    columns_to_index=["image"],
    
    # Vision configuration
    vision_provider="openai",
    vision_model="gpt-4o-mini",
    vision_prompt="Analyze this image: 1) Objects and colors 2) Setting 3) Text visible 4) Mood",
    vision_kwargs={
        "max_tokens": 300,
        "temperature": 0.3
    },
    
    # Optional: Add direct CLIP embeddings for image-to-image search
    use_clip=True,
    clip_model="openai/clip-vit-base-patch32"
)

# Insert images with automatic vision analysis
memory.insert([
    {"image": "sunset.jpg", "title": "Beach Sunset"},
    {"image": "city.jpg", "title": "Urban Skyline"}
])

# Search using natural language - works on automatically generated descriptions
query = "orange and pink colors in outdoor setting"
similarity = memory.image_description.similarity(query)
results = memory.where(similarity >= 0.4).order_by(similarity, asc=False).collect()
```

### Multimodal Memory (Text + Images)

```python
# Combined text and image memory for comprehensive search
schema = {
    "content_type": pxt.String,  # "text" or "image"
    "image": pxt.Image,
    "text_content": pxt.String,
    "title": pxt.String,
}

memory = Memory(
    namespace="multimodal",
    table_name="content_library",
    schema=schema,
    columns_to_index=["image", "text_content"],  # Index both modalities
    vision_prompt="Describe this image for search: key objects, colors, setting, and context."
)

# Insert mixed content
memory.insert([
    {
        "content_type": "image",
        "image": "garden.jpg",
        "text_content": None,
        "title": "Garden Photo"
    },
    {
        "content_type": "text", 
        "image": None,
        "text_content": "Guide to growing colorful flowers in small spaces",
        "title": "Gardening Tips"
    }
])

# Search across both text and images
query = "colorful flower garden"

# Find text matches
text_sim = memory.text_content.similarity(query)
text_results = memory.where(text_sim >= 0.3).select(memory.title, text_sim).collect()

# Find image matches
image_sim = memory.image_description.similarity(query) 
image_results = memory.where(image_sim >= 0.3).select(memory.title, image_sim).collect()

print("Text matches:", text_results)
print("Image matches:", image_results)
```

### Vision Provider Options

Choose from multiple vision providers:

```python
# OpenAI GPT-4o (default)
memory = Memory(
    schema={"image": pxt.Image},
    columns_to_index=["image"],
    vision_provider="openai",
    vision_model="gpt-4o-mini",
    vision_kwargs={"max_tokens": 200}
)

# Google Gemini
memory = Memory(
    schema={"image": pxt.Image},
    columns_to_index=["image"],
    vision_provider="gemini", 
    vision_model="gemini-2.0-flash",
    vision_kwargs={
        "config": {
            "temperature": 0.4,
            "max_output_tokens": 300
        }
    }
)

# Anthropic Claude
memory = Memory(
    schema={"image": pxt.Image},
    columns_to_index=["image"],
    vision_provider="anthropic",
    vision_model="claude-3-haiku-20240307"
)
```

The image descriptions are automatically generated and indexed when you insert data, enabling powerful semantic search across your visual content without any additional setup.
