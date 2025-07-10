from pixelmemory import Memory
import pixeltable as pxt

# RAG (Retrieval-Augmented Generation) memory for LLM context enhancement
schema = {
    "document_id": pxt.String,
    "chunk_id": pxt.String,
    "content": pxt.String,
    "metadata": pxt.Json,
    "document_type": pxt.String,
    "relevance_score": pxt.Float,
    "last_retrieved": pxt.String,
}

mem = Memory(
    namespace="rag_system",
    table_name="document_chunks",
    schema=schema,
    columns_to_index=["content"],
    text_embedding_model="all-MiniLM-L6-v2",
    if_exists="replace",
)

# Store document chunks for RAG retrieval
document_chunks = [
    {
        "document_id": "python_guide_001",
        "chunk_id": "chunk_001",
        "content": "Python list comprehensions provide a concise way to create lists. They consist of brackets containing an expression followed by a for clause, then zero or more for or if clauses.",
        "metadata": {
            "section": "List Comprehensions",
            "page": 45,
            "difficulty": "intermediate",
        },
        "document_type": "tutorial",
        "relevance_score": 0.0,
        "last_retrieved": "never",
    },
    {
        "document_id": "python_guide_001",
        "chunk_id": "chunk_002",
        "content": "Exception handling in Python uses try-except blocks. The try block contains risky code, except blocks handle specific exceptions, and finally blocks run cleanup code.",
        "metadata": {"section": "Error Handling", "page": 67, "difficulty": "beginner"},
        "document_type": "tutorial",
        "relevance_score": 0.0,
        "last_retrieved": "never",
    },
    {
        "document_id": "api_docs_002",
        "chunk_id": "chunk_003",
        "content": "REST API design follows stateless principles. Each request contains all information needed to process it. Use HTTP methods (GET, POST, PUT, DELETE) semantically.",
        "metadata": {
            "section": "REST Principles",
            "version": "v2.1",
            "category": "architecture",
        },
        "document_type": "documentation",
        "relevance_score": 0.0,
        "last_retrieved": "never",
    },
    {
        "document_id": "best_practices_003",
        "chunk_id": "chunk_004",
        "content": "Code review best practices include checking for security vulnerabilities, performance issues, maintainability, and adherence to coding standards.",
        "metadata": {"topic": "Code Quality", "author": "dev_team", "priority": "high"},
        "document_type": "guidelines",
        "relevance_score": 0.0,
        "last_retrieved": "never",
    },
]

mem.insert(document_chunks)


# RAG Pattern 1: Basic semantic retrieval for LLM context
def retrieve_context_for_llm(query, top_k=3, min_relevance=0.3):
    """Retrieve relevant document chunks for LLM context augmentation"""
    print(f"RAG retrieval for query: '{query}'")

    similarity = mem.content.similarity(query)

    # Get most relevant chunks
    relevant_chunks = (
        mem.where(similarity >= min_relevance)
        .order_by(similarity, asc=False)
        .select(mem.content, mem.metadata, mem.document_type, relevance=similarity)
        .limit(top_k)
        .collect()
    )

    return relevant_chunks


# Example RAG retrieval
user_query = "How do I handle errors in Python?"
context_chunks = retrieve_context_for_llm(user_query)
print("Retrieved context for LLM:")
for chunk in context_chunks:
    print(f"- Relevance: {chunk[3]:.3f}")
    print(f"  Content: {chunk[0][:100]}...")
    print(f"  Type: {chunk[2]}")
    print()

# RAG Pattern 2: Filtered retrieval by document type
print("Retrieval filtered by documentation type:")
doc_query = "API design principles"
doc_similarity = mem.content.similarity(doc_query)

filtered_context = (
    mem.where(
        (mem.document_type == "documentation") | (mem.document_type == "guidelines")
    )
    .order_by(doc_similarity, asc=False)
    .select(mem.content, mem.document_id, mem.document_type, relevance=doc_similarity)
    .limit(2)
    .collect()
)
print(filtered_context)

# RAG Pattern 3: Metadata-enhanced retrieval
print("\nDifficulty-based retrieval for beginner-friendly content:")
beginner_query = "Python error handling basics"
beginner_similarity = mem.content.similarity(beginner_query)

beginner_content = (
    mem.where(mem.metadata.contains('"difficulty": "beginner"'))
    .order_by(beginner_similarity, asc=False)
    .select(mem.content, mem.metadata, relevance=beginner_similarity)
    .collect()
)
print(beginner_content)

# RAG Pattern 4: Multi-turn conversation context building
conversation_context = []


def get_conversational_rag_context(query, conversation_history=None, max_chunks=4):
    """Build context considering conversation history"""
    print(f"\nConversational RAG for: '{query}'")

    # Combine current query with recent conversation for context
    if conversation_history:
        expanded_query = f"{query} {' '.join(conversation_history[-2:])}"
    else:
        expanded_query = query

    similarity = mem.content.similarity(expanded_query)

    context = (
        mem.order_by(similarity, asc=False)
        .select(mem.content, mem.document_id, mem.chunk_id, relevance=similarity)
        .limit(max_chunks)
        .collect()
    )

    return context


# Simulate multi-turn conversation
conversation_history = [
    "How do I handle errors in Python?",
    "What about list comprehensions?",
]
current_query = "Can you show me examples?"

conversational_context = get_conversational_rag_context(
    current_query, conversation_history
)
print("Conversational context:", conversational_context)


# RAG Pattern 5: Hybrid retrieval with re-ranking
def hybrid_rag_retrieval(query, boost_recent=True, top_k=3):
    """Advanced RAG with multiple ranking factors"""
    print(f"\nHybrid RAG retrieval for: '{query}'")

    content_sim = mem.content.similarity(query)

    # Boost recently retrieved content (simulated)
    retrieval_boost = mem.last_retrieved != "never"

    hybrid_context = (
        mem.select(
            mem.content,
            mem.metadata,
            mem.document_type,
            content_relevance=content_sim,
            recency_boost=retrieval_boost,
            hybrid_score=content_sim + (retrieval_boost * 0.1 if boost_recent else 0),
        )
        .order_by(
            content_sim + (retrieval_boost * 0.1 if boost_recent else 0), asc=False
        )
        .limit(top_k)
        .collect()
    )

    return hybrid_context


# Example hybrid retrieval
hybrid_results = hybrid_rag_retrieval("Python programming best practices")
print("Hybrid RAG results:")
for result in hybrid_results:
    print(f"- Content: {result[0][:80]}...")
    print(f"  Type: {result[2]}, Relevance: {result[3]:.3f}")
