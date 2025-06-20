from pixelmemory import Memory
import pixeltable as pxt

# Conversation memory for LLM agents
schema = {
    "user_id": pxt.String,
    "message": pxt.String,
    "role": pxt.String,  # 'user', 'assistant', 'system'
    "timestamp": pxt.String,
    "conversation_id": pxt.String,
    "intent": pxt.String,
    "sentiment": pxt.String
}

mem = Memory(
    namespace="agent_conversations",
    table_name="chat_history",
    schema=schema,
    columns_to_index=["message"],
    if_exists="replace"
)

# Store conversation history
conversations = [
    {"user_id": "user_123", "message": "I need help with Python coding", "role": "user", "timestamp": "2024-01-15T10:00:00", "conversation_id": "conv_001", "intent": "technical_help", "sentiment": "neutral"},
    {"user_id": "agent", "message": "I'd be happy to help you with Python! What specific topic are you working on?", "role": "assistant", "timestamp": "2024-01-15T10:00:30", "conversation_id": "conv_001", "intent": "assistance", "sentiment": "positive"},
    {"user_id": "user_123", "message": "How do I handle exceptions in Python?", "role": "user", "timestamp": "2024-01-15T10:01:00", "conversation_id": "conv_001", "intent": "technical_help", "sentiment": "neutral"},
    {"user_id": "user_456", "message": "What's the weather like today?", "role": "user", "timestamp": "2024-01-15T11:00:00", "conversation_id": "conv_002", "intent": "weather_query", "sentiment": "neutral"},
    {"user_id": "agent", "message": "I don't have access to real-time weather data, but I can help you find weather information sources.", "role": "assistant", "timestamp": "2024-01-15T11:00:30", "conversation_id": "conv_002", "intent": "information", "sentiment": "helpful"}
]

mem.insert(conversations)

# Retrieval Pattern 1: Get recent conversation context for a user
print("Recent conversation context for user_123:")
user_context = (
    mem.where(mem.user_id == "user_123")
    .select(mem.role, mem.message, mem.timestamp)
    .order_by(mem.timestamp, asc=False)
    .limit(5)
    .collect()
)
print(user_context)

# Retrieval Pattern 2: Semantic search for similar user queries
print("\nSimilar queries to 'programming help':")
similarity = mem.message.similarity("programming help")
similar_queries = (
    mem.where(mem.role == "user")
    .order_by(similarity, asc=False)
    .select(mem.message, mem.intent, similarity=similarity)
    .limit(3)
    .collect()
)
print(similar_queries)

# Retrieval Pattern 3: Intent-based retrieval for agent responses
print("\nPrevious technical help responses:")
tech_responses = (
    mem.where((mem.intent == "technical_help") & (mem.role == "assistant"))
    .select(mem.message, mem.conversation_id)
    .collect()
)
print(tech_responses)

# Retrieval Pattern 4: Conversation thread reconstruction
print("\nFull conversation thread for conv_001:")
conversation_thread = (
    mem.where(mem.conversation_id == "conv_001")
    .select(mem.role, mem.message, mem.timestamp)
    .order_by(mem.timestamp, asc=True)
    .collect()
)
print(conversation_thread)

# Agent memory retrieval for context-aware responses
print("\nContext retrieval for next response:")
current_user = "user_123"
current_intent = "technical_help"

# Get user's conversation history + similar past interactions
context_query = (
    mem.where(
        (mem.user_id == current_user) | 
        ((mem.intent == current_intent) & (mem.role == "assistant"))
    )
    .select(mem.role, mem.message, mem.intent)
    .limit(10)
    .collect()
)
print(f"Context for {current_user} with {current_intent} intent:", context_query)
