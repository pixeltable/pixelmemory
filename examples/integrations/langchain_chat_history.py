# pip install langchain-openai langchain pixeltable
import pixeltable as pxt
from pixelmemory import Memory
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

# 1. Set up PixelMemory as a persistent message store
mem = Memory(
    namespace="langchain_memory",
    table_name="chat_history",
    schema={'session_id': pxt.String, 'messages': pxt.Json},
    if_exists="replace_force"
)

# 2. Initialize the LangChain model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# 3. Create a simple chat function with persistent memory
def chat(session_id: str, user_input: str, system_prompt: str = "You are a helpful assistant.", limit: int = 10) -> AIMessage:
    """Handles a chat turn, persisting history in PixelMemory."""
    # Load history; start with a system message if new session
    history = mem.where(mem.session_id == session_id).select(mem.messages).limit(limit).collect()
    messages = messages_from_dict(history[0]['messages']) if history else [SystemMessage(system_prompt)]
    
    # Add user's message and invoke the model
    messages.append(HumanMessage(user_input))
    ai_response = model.invoke(messages)
    messages.append(ai_response)
    
    # Save the updated history back to PixelMemory
    mem.insert([{'session_id': session_id, 'messages': messages_to_dict(messages)}])
    
    return ai_response

# 4. Demonstrate the conversation
SESSION_ID = "user_123"
print("--- Conversation Turn 1 ---")
response1 = chat(SESSION_ID, "Hi! I'm Bob and I live in Paris.")
print(f"AI: {response1.content}")

print("\n--- Conversation Turn 2 (Memory in action) ---")
response2 = chat(SESSION_ID, "What's my name and where do I live?")
print(f"AI: {response2.content}")

# Verify that a new session is independent
print("\n--- New Conversation (Session user_456) ---")
response3 = chat("user_456", "What's my name?")
print(f"AI: {response3.content}")
