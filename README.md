# Pixelmemory

**Reference Implementation: Multimodal Memory Layer Built on [Pixeltable](https://github.com/pixeltable/pixeltable)**

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Package](https://img.shields.io/pypi/v/pixelmemory?color=4D148C)](https://pypi.org/project/pixelmemory/)
[![Discord](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)

## Overview

Pixelmemory demonstrates how to build sophisticated memory layers using [Pixeltable's](https://github.com/pixeltable/pixeltable) declarative data infrastructure. This reference implementation shows how to create persistent, searchable, multimodal memory for stateful AI agents.

```mermaid
graph TB
    A[Your AI Agent] --> B[Pixelmemory Layer]
    B --> C[Pixeltable Foundation]
    C --> D[(Local Storage)]
    C --> E[Vector Indexes]
    C --> F[Computed Columns]
    
    B --> G[Text Memory]
    B --> H[Image Memory]
    B --> I[Video Memory]
    B --> J[Audio Memory]
    B --> K[Document Memory]
    
    G --> L[Semantic Search]
    H --> L
    I --> L
    J --> L
    K --> L
```

## Why Use Pixelmemory?

Most AI applications today are stateless - they forget everything between sessions. Pixelmemory solves this by providing:

- **Persistent memory** that survives between sessions
- **Semantic search** across all data types  
- **Local-first storage** with no vendor lock-in
- **Multimodal support** for text, images, videos, audio, documents
- **Production-ready** foundation built on Pixeltable

## Installation

```bash
pip install pixelmemory
```

*Note: Pixeltable is automatically installed as a dependency - no additional setup required.*

## Quick Start

```python
from pixelmemory import Memory
from pixelmemory.context import Text

# Define memory structure
context = [
    Text(id="content", embed=True),    # Searchable content
    Text(id="user_id", embed=False),   # Metadata
]

# Create memory instance
memory = Memory(context=context, namespace="chatbot")

# Add memories
entry = memory.Entry(content="I love Python programming", user_id="user_123")
memory.add(entry)

# Semantic search
similarity = memory.content.similarity("programming languages")
results = memory.where(similarity >= 0.5).collect()
```

## Architecture

```mermaid
sequenceDiagram
    participant App as Your App
    participant PM as Pixelmemory
    participant PT as Pixeltable
    participant DB as Pixeltable Storage
    
    App->>PM: Create memory with context
    PM->>PT: Create table with schema
    PT->>DB: Initialize table & indexes
    
    App->>PM: Add memory entry
    PM->>PT: Insert with computed columns
    PT->>DB: Store data + embeddings
    
    App->>PM: Search memories
    PM->>PT: Query with similarity
    PT->>DB: Vector search
    DB-->>App: Ranked results
```

## Examples

### Basic Text Memory
```python
from pixelmemory import Memory
from pixelmemory.context import Text

context = [Text(id="content", embed=True)]
memory = Memory(context=context)

# Add and search
entry = memory.Entry(content="Learning about AI")
memory.add(entry)

# Semantic search
sim = memory.content.similarity("artificial intelligence")
results = memory.where(sim >= 0.3).collect()
```

### Multimodal Memory
```python
from pixelmemory.context import Text, Image, Video

# Support multiple data types
context = [
    Text(id="description", embed=True),
    Image(id="screenshot", provider="openai", model="gpt-4o-mini"),
    Video(id="demo_video"),
]

memory = Memory(context=context, namespace="multimodal_app")
```

### Integration with LangChain
```python
from pixelmemory import Memory
from pixelmemory.context import Text
from langchain.chat_models import init_chat_model

# Persistent chat memory
context = [
    Text(id="session_id", embed=False),
    Text(id="messages", embed=False),
]

memory = Memory(context=context, namespace="langchain_chat")
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Chat with memory
def chat_with_memory(session_id: str, message: str):
    # Retrieve history, generate response, save back to memory
    # (See examples/integrations/langchain_chat_history.py)
    pass
```

## Example Files

All examples use the context-based API and are ready to run:

**Basic Examples** (no external dependencies):
- `examples/getting_started/01_basic_memory.py`
- `examples/multimodal/text.py`

**Multimodal Examples** (require OpenAI API key):
- `examples/multimodal/images.py`
- `examples/multimodal/video.py`
- `examples/multimodal/audio.py`

**Integration Examples**:
- `examples/integrations/langchain_chat_history.py` (requires OpenAI API key)
- `examples/integrations/crewai_agentic_rag.py` (requires `pip install crewai`)
- `examples/fastapi/memory_service.py` (requires `pip install fastapi uvicorn`)

## Built on Pixeltable

Pixelmemory is powered by **[Pixeltable](https://github.com/pixeltable/pixeltable)** - the leading open-source platform for multimodal AI applications.

### Why Pixeltable?

- **806+ GitHub stars** and active community
- **Declarative data infrastructure** for AI workloads
- **Built-in multimodal support** (images, videos, audio, documents)
- **Automatic embedding indexes** and vector search
- **Incremental computation** - only recomputes what's changed
- **Production-ready** with robust data persistence

```mermaid
graph LR
    A[Pixelmemory] --> B[Pixeltable Platform]
    B --> C[Tables & Schemas]
    B --> D[Computed Columns]
    B --> E[Embedding Indexes]
    B --> F[Vector Search]
    B --> G[Multimodal Support]
    
    H[Your Use Case] --> A
    I[Other AI Apps] --> B
    J[RAG Systems] --> B
    K[Computer Vision] --> B
```

## Next Steps

**Ready to build more advanced AI applications?**

1. **[Explore Pixeltable](https://github.com/pixeltable/pixeltable)** - Master the underlying platform (800+ stars)
2. **[Read the documentation](https://docs.pixeltable.com/)** - Comprehensive guides and tutorials  
3. **[Join the community](https://discord.gg/QPyqFYx2UN)** - Get help and share your implementations
4. **[See advanced examples](https://docs.pixeltable.com/docs/examples/use-cases)** - RAG, computer vision, audio processing

**Learn more about building stateful agents**: [Building Memory-Powered AI: Creating Stateful Agents with Pixeltable](https://www.pixeltable.com/blog/building-memory-powered-ai-stateful-agents-pixeltable)

---

**Remember**: Pixelmemory is a reference implementation. Use it as inspiration to build your own memory architecture using [Pixeltable's](https://github.com/pixeltable/pixeltable) flexible primitives.

## License

Apache 2.0 License