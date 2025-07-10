from pixelmemory import Memory
import pixeltable as pxt

# Knowledge base for agent fact retrieval and question answering
schema = {
    "topic": pxt.String,
    "content": pxt.String,
    "source": pxt.String,
    "category": pxt.String,
    "confidence": pxt.Float,
    "last_accessed": pxt.String,
    "keywords": pxt.Json,
}

mem = Memory(
    namespace="agent_knowledge",
    table_name="facts_and_procedures",
    schema=schema,
    columns_to_index=["content", "topic"],
    if_exists="replace",
)

# Store knowledge base entries for agent retrieval
knowledge_entries = [
    {
        "topic": "Python Exception Handling",
        "content": "Use try-except blocks to handle exceptions in Python. The try block contains code that might raise an exception, and except blocks handle specific exception types.",
        "source": "Python Documentation",
        "category": "programming",
        "confidence": 0.95,
        "last_accessed": "2024-01-15T10:00:00",
        "keywords": ["python", "exceptions", "error handling", "try-except"],
    },
    {
        "topic": "API Rate Limiting",
        "content": "Implement exponential backoff when hitting API rate limits. Start with 1 second delay, double on each retry, and include jitter to avoid thundering herd.",
        "source": "Best Practices Guide",
        "category": "api_design",
        "confidence": 0.88,
        "last_accessed": "2024-01-15T09:30:00",
        "keywords": ["api", "rate limiting", "backoff", "retry logic"],
    },
    {
        "topic": "Database Connection Pooling",
        "content": "Connection pooling reduces overhead by reusing database connections. Configure pool size based on concurrent users and database capacity.",
        "source": "Database Optimization Manual",
        "category": "database",
        "confidence": 0.92,
        "last_accessed": "2024-01-14T16:20:00",
        "keywords": ["database", "connection pool", "performance", "optimization"],
    },
]

mem.insert(knowledge_entries)

# Retrieval Pattern 1: Semantic search for agent Q&A
user_question = "How should I handle errors in my Python code?"
print(f"User question: {user_question}")
print("Relevant knowledge retrieval:")

similarity = mem.content.similarity(user_question)
relevant_knowledge = (
    mem.order_by(similarity, asc=False)
    .select(mem.topic, mem.content, mem.confidence, relevance_score=similarity)
    .limit(2)
    .collect()
)
print(relevant_knowledge)

# Retrieval Pattern 2: Category-based knowledge lookup
print("\nProgramming-related knowledge for context:")
programming_context = (
    mem.where(mem.category == "programming")
    .select(mem.topic, mem.content, mem.keywords)
    .collect()
)
print(programming_context)

# Retrieval Pattern 3: Multi-field semantic search
query = "database performance issues"
print(f"\nMulti-field search for: {query}")

topic_match = mem.topic.similarity(query)
content_match = mem.content.similarity(query)

multi_field_results = (
    mem.select(
        mem.topic,
        mem.category,
        topic_relevance=topic_match,
        content_relevance=content_match,
        combined_score=(topic_match + content_match) / 2,
    )
    .order_by((topic_match + content_match) / 2, asc=False)
    .limit(3)
    .collect()
)
print(multi_field_results)

# Retrieval Pattern 4: Confidence-weighted retrieval
print("\nHigh-confidence knowledge (>0.9) for reliable agent responses:")
high_confidence = (
    mem.where(mem.confidence > 0.9)
    .select(mem.topic, mem.content, mem.confidence, mem.source)
    .order_by(mem.confidence, asc=False)
    .collect()
)
print(high_confidence)


# Agent workflow: Retrieve context for generating response
def get_agent_context(user_query, category_filter=None, min_confidence=0.8):
    """Retrieve relevant context for agent response generation"""
    print(f"\nAgent context retrieval for: '{user_query}'")

    # Build query with semantic search and filters
    similarity = mem.content.similarity(user_query)
    query = mem.where(mem.confidence >= min_confidence)

    if category_filter:
        query = query.where(mem.category == category_filter)

    context = (
        query.order_by(similarity, asc=False)
        .select(
            mem.topic,
            mem.content,
            mem.source,
            mem.confidence,
            similarity_score=similarity,
        )
        .limit(3)
        .collect()
    )

    return context


# Example agent context retrieval
context = get_agent_context("API integration best practices", min_confidence=0.85)
print("Retrieved context:", context)
