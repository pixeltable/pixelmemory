from pixelmemory import Memory
import pixeltable as pxt

# Agent learning memory for behavioral adaptation and improvement
schema = {
    "user_id": pxt.String,
    "interaction_type": pxt.String,
    "user_input": pxt.String,
    "agent_response": pxt.String,
    "user_feedback": pxt.String,  # 'positive', 'negative', 'neutral'
    "feedback_score": pxt.Float,  # 1-5 rating
    "response_effectiveness": pxt.Float,
    "timestamp": pxt.String,
    "context_tags": pxt.Json,
    "learned_pattern": pxt.String
}

mem = Memory(
    namespace="agent_learning",
    table_name="behavioral_patterns",
    schema=schema,
    columns_to_index=["user_input", "agent_response", "learned_pattern"],
    if_exists="replace"
)

# Store agent learning data for behavioral improvement
learning_data = [
    {
        "user_id": "user_001",
        "interaction_type": "technical_question",
        "user_input": "How do I optimize database queries?",
        "agent_response": "Here are several database optimization techniques: indexing, query analysis, and connection pooling.",
        "user_feedback": "positive",
        "feedback_score": 4.5,
        "response_effectiveness": 0.9,
        "timestamp": "2024-01-15T10:00:00",
        "context_tags": ["database", "performance", "optimization"],
        "learned_pattern": "Users prefer detailed technical explanations with multiple approaches"
    },
    {
        "user_id": "user_002", 
        "interaction_type": "code_review",
        "user_input": "Can you review this Python function?",
        "agent_response": "Your function looks good, but consider adding error handling and type hints.",
        "user_feedback": "neutral",
        "feedback_score": 3.0,
        "response_effectiveness": 0.6,
        "timestamp": "2024-01-15T10:30:00",
        "context_tags": ["code_review", "python", "suggestions"],
        "learned_pattern": "Code reviews need specific examples and actionable feedback"
    },
    {
        "user_id": "user_001",
        "interaction_type": "explanation",
        "user_input": "Explain async programming simply",
        "agent_response": "Async programming is like cooking multiple dishes - you start one, switch to another while waiting, maximizing efficiency.",
        "user_feedback": "positive",
        "feedback_score": 5.0,
        "response_effectiveness": 0.95,
        "timestamp": "2024-01-15T11:00:00",
        "context_tags": ["async", "analogies", "simple_explanation"],
        "learned_pattern": "Analogies make complex concepts more accessible"
    },
    {
        "user_id": "user_003",
        "interaction_type": "troubleshooting",
        "user_input": "My API is returning 500 errors",
        "agent_response": "Let's check your logs, database connections, and recent deployments systematically.",
        "user_feedback": "positive", 
        "feedback_score": 4.8,
        "response_effectiveness": 0.92,
        "timestamp": "2024-01-15T11:30:00",
        "context_tags": ["debugging", "systematic_approach", "API"],
        "learned_pattern": "Structured troubleshooting approach builds user confidence"
    }
]

mem.insert(learning_data)

# Learning Pattern 1: Identify successful response patterns
print("High-performing response patterns:")
successful_patterns = (
    mem.where(mem.feedback_score >= 4.0)
    .select(
        mem.interaction_type,
        mem.learned_pattern,
        mem.feedback_score,
        mem.response_effectiveness
    )
    .order_by(mem.feedback_score, asc=False)
    .collect()
)
print(successful_patterns)

# Learning Pattern 2: User-specific preference learning
def learn_user_preferences(user_id):
    """Learn what response styles work best for specific users"""
    print(f"\nLearning preferences for {user_id}:")
    
    user_patterns = (
        mem.where(mem.user_id == user_id)
        .select(
            mem.interaction_type,
            mem.learned_pattern,
            avg_score=mem.feedback_score.avg(),
            avg_effectiveness=mem.response_effectiveness.avg(),
            interaction_count=mem.user_input.count()
        )
        .order_by(mem.feedback_score.avg(), asc=False)
        .collect()
    )
    
    return user_patterns

user_prefs = learn_user_preferences("user_001")
print("User preferences:", user_prefs)

# Learning Pattern 3: Response adaptation based on context
def get_adaptive_response_guidance(current_input, interaction_type):
    """Get guidance for response style based on learned patterns"""
    print(f"\nAdaptive guidance for: '{current_input}' ({interaction_type})")
    
    # Find similar successful interactions
    input_similarity = mem.user_input.similarity(current_input)
    
    guidance = (
        mem.where(
            (mem.interaction_type == interaction_type) & 
            (mem.feedback_score >= 4.0)
        )
        .order_by(input_similarity, asc=False)
        .select(
            mem.learned_pattern,
            mem.response_effectiveness,
            mem.context_tags,
            similarity=input_similarity
        )
        .limit(3)
        .collect()
    )
    
    return guidance

# Example adaptive guidance
guidance = get_adaptive_response_guidance(
    "How do I handle errors in my code?", 
    "technical_question"
)
print("Adaptive guidance:", guidance)

# Learning Pattern 4: Identify improvement opportunities
print("\nLow-performing patterns for improvement:")
improvement_areas = (
    mem.where(mem.feedback_score < 3.5)
    .select(
        mem.interaction_type,
        mem.user_input,
        mem.agent_response,
        mem.feedback_score,
        mem.user_feedback
    )
    .collect()
)
print("Areas needing improvement:", improvement_areas)

# Learning Pattern 5: Trend analysis for behavioral evolution
print("\nResponse effectiveness trends by interaction type:")
effectiveness_trends = (
    mem.select(
        mem.interaction_type,
        avg_effectiveness=mem.response_effectiveness.avg(),
        avg_score=mem.feedback_score.avg(),
        total_interactions=mem.user_input.count()
    )
    .collect()
)
print("Effectiveness trends:", effectiveness_trends)

# Learning Pattern 6: Context-aware pattern matching
def find_contextual_patterns(context_tags, min_effectiveness=0.8):
    """Find successful patterns for specific contexts"""
    print(f"\nSuccessful patterns for context: {context_tags}")
    
    # Find interactions with overlapping context tags
    contextual_patterns = []
    for tag in context_tags:
        tag_patterns = (
            mem.where(
                (mem.context_tags.contains(tag)) & 
                (mem.response_effectiveness >= min_effectiveness)
            )
            .select(
                mem.learned_pattern,
                mem.response_effectiveness,
                mem.feedback_score,
                mem.context_tags
            )
            .collect()
        )
        contextual_patterns.extend(tag_patterns)
    
    return contextual_patterns

# Example contextual pattern matching
context_patterns = find_contextual_patterns(["database", "optimization"])
print("Contextual patterns:", context_patterns)

# Agent self-reflection: Analyze learning progress
print("\nAgent self-reflection analysis:")
learning_stats = (
    mem.select(
        total_interactions=mem.user_input.count(),
        avg_user_satisfaction=mem.feedback_score.avg(),
        avg_effectiveness=mem.response_effectiveness.avg(),
        positive_feedback_rate=(mem.user_feedback == 'positive').sum() / mem.user_input.count()
    )
    .collect()
)
print("Learning statistics:", learning_stats)

# Continuous improvement: Identify next learning priorities
print("\nNext learning priorities:")
learning_priorities = (
    mem.where(mem.feedback_score < 4.0)
    .select(
        mem.interaction_type,
        improvement_needed=4.0 - mem.feedback_score.avg(),
        frequency=mem.user_input.count()
    )
    .order_by(mem.user_input.count(), asc=False)
    .collect()
)
print("Priority areas:", learning_priorities)
