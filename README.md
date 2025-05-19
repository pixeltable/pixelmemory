# Pixelmemory

A memory management layer for AI agents and LLMs, built on the powerful pixeltable database.

## Introduction

Pixelmemory provides semantic memory capabilities for AI applications, enabling long-term memory for conversational agents, RAG systems, and AI assistants. Built on pixeltable's vector database capabilities, it offers efficient storage and retrieval of conversation history and contextual information.

## Installation

```bash
pip install pixelmemory
```

## Key Features

- **Semantic Memory Retrieval**: Find relevant past conversations using vector similarity search
- **Multi-User Support**: Maintain separate memory spaces for different users
- **Metadata Storage**: Store and retrieve structured data alongside memories
- **Pixeltable Integration**: Leverages pixeltable's powerful vector database capabilities

## Use Cases

- **AI Memory Systems**: Add long-term memory to conversational AI agents
- **Personalized AI Assistants**: Remember user preferences and past interactions
- **Knowledge Management**: Store and retrieve information from past conversations
- **Agent Memory**: Provide context for autonomous AI agents

## Quickstart Guide

### Installation

```bash
pip install pixelmemory
```

### Basic Usage with Anthropic Claude

```python
from anthropic import Anthropic
from pixelmemory import Memory

# Initialize Anthropic client and Memory
anthropic_client = Anthropic()
memory = Memory()

# Example: Chat with memories
def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # Generate Assistant response
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
    
    # Create Anthropic message format
    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": message}]
    )
    assistant_response = response.content[0].text

    # Create new memories from the conversation
    memory_messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_response}
    ]
    memory.add(memory_messages, user_id=user_id)

    return assistant_response

# Example: Interactive chat
def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()
```

## How It Works

Pixelmemory uses pixeltable's vector database capabilities to store and retrieve memories:

1. **Memory Storage**: Conversations are stored with vector embeddings for semantic search
2. **Memory Retrieval**: When a new query comes in, similar past conversations are retrieved
3. **Context Integration**: Retrieved memories are included in the prompt to the LLM
4. **Continuous Learning**: New conversations are stored for future reference

## Advanced Features

- **Custom Embedding Models**: Use your preferred embedding model for specialized domains
- **Flexible Querying**: Search by semantic similarity, time ranges, or custom filters
- **Metadata Management**: Store and retrieve structured data alongside memories
- **Multi-User Architecture**: Maintain separate memory spaces for different users

## Integrating with AI Memory Systems

Pixelmemory can be used as a drop-in replacement for other memory systems in AI applications. It provides a simple API for storing and retrieving memories, making it easy to integrate with existing AI systems.

Keywords: AI memory, long-term memory, semantic memory, conversation memory, agent memory, AI assistant memory, memory retrieval, memory storage, vector database, pixeltable, AI memory systems, conversational memory

## Requirements

- Python 3.9+
- pixeltable 0.3.8+

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
