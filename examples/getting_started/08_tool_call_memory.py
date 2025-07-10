from pixelmemory import Memory
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

# Tool call memory for LLM function calling and agent actions
schema = {
    "user_id": pxt.String,
    "tool_name": pxt.String,
    "parameters": pxt.Json,
    "result": pxt.String,
    "success": pxt.Bool,
    "timestamp": pxt.String,
    "context": pxt.String,
    "execution_time": pxt.Float,
}

mem = Memory(
    namespace="agent_tools",
    table_name="function_calls",
    schema=schema,
    columns_to_index=["context", "result"],
    if_exists="replace",
)

# Store tool call history for learning and optimization
tool_calls = [
    {
        "user_id": "user_123",
        "tool_name": "get_weather",
        "parameters": {"location": "San Francisco", "units": "celsius"},
        "result": "The weather in San Francisco is 18°C and cloudy",
        "success": True,
        "timestamp": "2024-01-15T10:00:00",
        "context": "User asked about weather for planning outdoor activities",
        "execution_time": 1.2,
    },
    {
        "user_id": "user_123",
        "tool_name": "send_email",
        "parameters": {"to": "colleague@company.com", "subject": "Meeting reschedule"},
        "result": "Email sent successfully",
        "success": True,
        "timestamp": "2024-01-15T10:05:00",
        "context": "User needs to reschedule meeting due to weather",
        "execution_time": 2.1,
    },
    {
        "user_id": "user_456",
        "tool_name": "search_web",
        "parameters": {"query": "Python async programming", "max_results": 5},
        "result": "Found 5 relevant articles about Python asyncio and concurrent programming",
        "success": True,
        "timestamp": "2024-01-15T11:00:00",
        "context": "User learning about asynchronous programming concepts",
        "execution_time": 3.5,
    },
    {
        "user_id": "user_456",
        "tool_name": "code_analyzer",
        "parameters": {"language": "python", "code": "async def example(): pass"},
        "result": "Code analysis complete: Function is properly defined but empty",
        "success": True,
        "timestamp": "2024-01-15T11:02:00",
        "context": "User testing async function syntax",
        "execution_time": 0.8,
    },
]

mem.insert(tool_calls)

# Retrieval Pattern 1: User's tool usage history for personalization
print("Tool usage history for user_123:")
user_tools = (
    mem.where(mem.user_id == "user_123")
    .select(mem.tool_name, mem.parameters, mem.timestamp, mem.context)
    .order_by(mem.timestamp, asc=False)
    .collect()
)
print(user_tools)

# Retrieval Pattern 2: Similar contexts for tool recommendation
current_context = "User wants to learn about programming"
print(f"\nTool recommendations for context: '{current_context}'")

context_similarity = mem.context.similarity(current_context)
recommended_tools = (
    mem.order_by(context_similarity, asc=False)
    .select(
        mem.tool_name, mem.parameters, mem.context, similarity_score=context_similarity
    )
    .limit(3)
    .collect()
)
print(recommended_tools)

# Retrieval Pattern 3: Tool success patterns and optimization
print("\nTool performance analysis:")
tool_performance = (
    mem.group_by(mem.tool_name)
    .select(
        mem.tool_name,
        success_rate=(mem.success == True).sum() / mem.tool_name.count(),
        avg_execution_time=mem.execution_time.avg(),
        total_calls=mem.tool_name.count(),
    )
    .collect()
)
print(tool_performance)

# Retrieval Pattern 4: Sequential tool usage patterns
print("\nSequential tool patterns for workflow optimization:")
sequential_patterns = (
    mem.where(mem.user_id == "user_123")
    .select(mem.tool_name, mem.timestamp, mem.context)
    .order_by(mem.timestamp, asc=True)
    .collect()
)
print("Tool sequence for user_123:", sequential_patterns)


# Agent workflow: Get tool suggestions based on context and history
def suggest_tools_for_context(user_id, current_context, limit=3):
    """Suggest tools based on user history and similar contexts"""
    print(f"\nTool suggestions for {user_id} in context: '{current_context}'")

    # Get user's frequently used tools
    user_freq_tools = (
        mem.where(mem.user_id == user_id)
        .group_by(mem.tool_name)
        .select(
            mem.tool_name,
            success_rate=(mem.success == True).sum() / mem.tool_name.count(),
        )
        .order_by(mem.tool_name.count(), asc=False)
        .limit(limit)
        .collect()
    )

    # Get tools used in similar contexts by any user
    context_sim = mem.context.similarity(current_context)
    similar_context_tools = (
        mem.where(mem.success == True)
        .order_by(context_sim, asc=False)
        .select(mem.tool_name, mem.parameters, context_similarity=context_sim)
        .limit(limit)
        .collect()
    )

    return {
        "frequent_tools": user_freq_tools,
        "context_similar_tools": similar_context_tools,
    }


# Example tool suggestion
suggestions = suggest_tools_for_context(
    "user_456", "User needs help with Python development"
)
print("Tool suggestions:", suggestions)

# Retrieval Pattern 5: Failed tool calls for debugging
print("\nFailed tool calls for improvement:")
failed_calls = (
    mem.where(mem.success == False)
    .select(mem.tool_name, mem.parameters, mem.result, mem.context)
    .collect()
)
print("Failed calls:", failed_calls if failed_calls else "No failed calls found")
