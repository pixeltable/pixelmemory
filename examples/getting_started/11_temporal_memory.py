from pixelmemory import Memory
import pixeltable as pxt

# Temporal memory for time-based agent reasoning and context windows
schema = {
    "event_id": pxt.String,
    "user_id": pxt.String,
    "event_type": pxt.String,
    "content": pxt.String,
    "timestamp": pxt.Timestamp,
    "priority": pxt.String,  # 'high', 'medium', 'low'
    "status": pxt.String,  # 'active', 'completed', 'expired'
    "tags": pxt.Json,
    "expires_at": pxt.Timestamp,
}

mem = Memory(
    namespace="temporal_agent",
    table_name="time_events",
    schema=schema,
    columns_to_index=["content"],
    if_exists="replace",
)

# Store temporal events for time-aware agent reasoning
from datetime import datetime, timedelta

base_time = datetime(2024, 1, 15, 10, 0, 0)

temporal_events = [
    {
        "event_id": "evt_001",
        "user_id": "user_123",
        "event_type": "reminder",
        "content": "Schedule team meeting for project review",
        "timestamp": base_time,
        "priority": "high",
        "status": "active",
        "tags": ["meeting", "project", "team"],
        "expires_at": base_time + timedelta(days=7),
    },
    {
        "event_id": "evt_002",
        "user_id": "user_123",
        "event_type": "task",
        "content": "Complete code review for authentication module",
        "timestamp": base_time + timedelta(hours=1),
        "priority": "medium",
        "status": "active",
        "tags": ["code_review", "auth", "security"],
        "expires_at": base_time + timedelta(days=3),
    },
    {
        "event_id": "evt_003",
        "user_id": "user_456",
        "event_type": "note",
        "content": "User reported slow API response times in production",
        "timestamp": base_time + timedelta(hours=2),
        "priority": "high",
        "status": "active",
        "tags": ["performance", "api", "bug"],
        "expires_at": base_time + timedelta(days=1),
    },
    {
        "event_id": "evt_004",
        "user_id": "user_123",
        "event_type": "task",
        "content": "Research database optimization techniques",
        "timestamp": base_time - timedelta(days=2),
        "priority": "low",
        "status": "completed",
        "tags": ["database", "performance", "research"],
        "expires_at": base_time + timedelta(days=30),
    },
]

mem.insert(temporal_events)

# Temporal Pattern 1: Recent events within time window
print("Events from the last 24 hours:")
now = base_time + timedelta(hours=3)
last_24h = now - timedelta(days=1)

recent_events = (
    mem.where((mem.timestamp >= last_24h) & (mem.timestamp <= now))
    .select(mem.content, mem.timestamp, mem.priority, mem.status)
    .order_by(mem.timestamp, asc=False)
    .collect()
)
print(recent_events)

# Temporal Pattern 2: Active events expiring soon
print("\nActive events expiring within 2 days:")
expiry_threshold = now + timedelta(days=2)

expiring_soon = (
    mem.where(
        (mem.status == "active")
        & (mem.expires_at <= expiry_threshold)
        & (mem.expires_at >= now)
    )
    .select(mem.content, mem.priority, mem.expires_at, time_left=(mem.expires_at - now))
    .order_by(mem.expires_at, asc=True)
    .collect()
)
print(expiring_soon)

# Temporal Pattern 3: Complex logical filtering with time bounds
print("\nHigh priority events OR recent performance issues:")
complex_query = (
    mem.where(
        (mem.priority == "high")
        | ((mem.content.contains("performance")) & (mem.timestamp >= last_24h))
    )
    .select(mem.content, mem.priority, mem.timestamp, mem.tags)
    .order_by(mem.timestamp, asc=False)
    .collect()
)
print(complex_query)


# Temporal Pattern 4: Time-bounded semantic search
def temporal_semantic_search(query, hours_back=24, min_priority=None):
    """Search within time window with optional priority filter"""
    print(f"\nTemporal semantic search: '{query}' (last {hours_back}h)")

    time_boundary = now - timedelta(hours=hours_back)
    similarity = mem.content.similarity(query)

    # Build query with temporal and logical constraints
    base_query = mem.where(mem.timestamp >= time_boundary)

    if min_priority:
        priority_values = {"low": 1, "medium": 2, "high": 3}
        min_val = priority_values.get(min_priority, 1)

        if min_priority == "medium":
            base_query = base_query.where(
                (mem.priority == "medium") | (mem.priority == "high")
            )
        elif min_priority == "high":
            base_query = base_query.where(mem.priority == "high")

    results = (
        base_query.order_by(similarity, asc=False)
        .select(mem.content, mem.timestamp, mem.priority, relevance=similarity)
        .limit(3)
        .collect()
    )

    return results


# Example temporal semantic search
search_results = temporal_semantic_search(
    "code review tasks", hours_back=48, min_priority="medium"
)
print(search_results)

# Temporal Pattern 5: Rolling window analysis for trends
print("\nRolling window analysis - event patterns by hour:")
time_windows = [
    (base_time - timedelta(hours=1), base_time),
    (base_time, base_time + timedelta(hours=1)),
    (base_time + timedelta(hours=1), base_time + timedelta(hours=2)),
    (base_time + timedelta(hours=2), base_time + timedelta(hours=3)),
]

for i, (start, end) in enumerate(time_windows):
    window_events = (
        mem.where((mem.timestamp >= start) & (mem.timestamp < end))
        .select(
            event_count=mem.event_id.count(),
            high_priority_count=(mem.priority == "high").sum(),
            active_count=(mem.status == "active").sum(),
        )
        .collect()
    )
    print(f"Hour {i}: {window_events}")


# Temporal Pattern 6: Contextual memory with decay
def get_contextual_memory_with_decay(user_id, current_context, max_age_hours=72):
    """Retrieve user context with time-based relevance decay"""
    print(f"\nContextual memory with decay for {user_id}:")

    cutoff_time = now - timedelta(hours=max_age_hours)
    context_similarity = mem.content.similarity(current_context)

    # Calculate time decay factor (more recent = higher weight)
    time_diff_hours = (now - mem.timestamp) / timedelta(hours=1)
    decay_factor = 1.0 / (1.0 + time_diff_hours * 0.1)  # Exponential decay

    contextual_memory = (
        mem.where(
            (mem.user_id == user_id)
            & (mem.timestamp >= cutoff_time)
            & (mem.status != "expired")
        )
        .select(
            mem.content,
            mem.event_type,
            mem.timestamp,
            context_relevance=context_similarity,
            time_decay=decay_factor,
            weighted_score=context_similarity * decay_factor,
        )
        .order_by(context_similarity * decay_factor, asc=False)
        .limit(5)
        .collect()
    )

    return contextual_memory


# Example contextual memory with temporal decay
context_memory = get_contextual_memory_with_decay(
    "user_123", "project planning and team coordination"
)
print(context_memory)

# Temporal Pattern 7: Multi-condition temporal filtering
print(
    "\nComplex temporal query - Active high-priority items OR completed tasks from last week:"
)
last_week = now - timedelta(days=7)

multi_condition = (
    mem.where(
        # Active high-priority items
        ((mem.status == "active") & (mem.priority == "high"))
        |
        # OR completed tasks from last week
        ((mem.status == "completed") & (mem.timestamp >= last_week))
    )
    .select(
        mem.content,
        mem.status,
        mem.priority,
        mem.timestamp,
        age_hours=(now - mem.timestamp) / timedelta(hours=1),
    )
    .order_by(mem.timestamp, asc=False)
    .collect()
)
print(multi_condition)
