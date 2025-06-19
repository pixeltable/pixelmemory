import uuid
from datetime import datetime
from openai import OpenAI
import pixeltable as pxt
from pixelmemory import Memory

openai_client = OpenAI()

memory = Memory(
    namespace="openai",
    table_name="chat_history",
    schema={
        "memory_id": pxt.Required[pxt.String],
        "user_id": pxt.String,
        "role": pxt.String,
        "message_content": pxt.String,
        "insert_at": pxt.Timestamp,
    },
    columns_to_index=["message_content"],
    embedding_model="intfloat/e5-large-v2",
    if_exists="ignore",
)

def chat_with_memories(current_user_message: str, user_id: str = "default_user") -> str:

    sim_expr = memory.message_content.similarity(current_user_message)
    retrieved_memories_result = (
        memory.where(memory.user_id == user_id)
        .order_by(sim_expr, asc=False)
        .select(memory.role, memory.message_content, similarity_score=sim_expr)
        .limit(3)
        .collect()
    )

    print(f"\n--- Debug: For query '{current_user_message}' (user: {user_id}) ---")
    if retrieved_memories_result:
        print(f"Retrieved {len(retrieved_memories_result)} memories:")
        for i, entry in enumerate(retrieved_memories_result):
            print(f"  Memory {i+1}: Role='{entry.get('role')}', Content='{entry.get('message_content')}', Score={entry.get('similarity_score'):.4f}")
    else:
        print("No memories retrieved.")

    memories_str = ""
    if retrieved_memories_result:
        memories_for_prompt = [
            f"- {entry.get('role')}: {entry.get('message_content')}"
            for entry in retrieved_memories_result
        ]
        memories_str = "\n".join(memories_for_prompt)

    system_prompt_content = (
        f"You are a helpful AI. Answer the question based on the current query and "
        f"the following relevant memories (if any):\nUser Memories:\n{memories_str}"
    )

    print(f"System Prompt (for LLM):\n{system_prompt_content}")
    print("--- End Debug ---")

    llm_messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": current_user_message}
    ]
    
    response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=llm_messages)
    assistant_response_content = response.choices[0].message.content

    new_messages_to_store = [
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": user_id,
            "role": "user",
            "message_content": current_user_message,
            "insert_at": datetime.now(),
        },
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": user_id, 
            "role": "assistant",
            "message_content": assistant_response_content,
            "insert_at": datetime.now(),
        }
    ]
    
    memory.insert(new_messages_to_store) 

    return assistant_response_content

def main():
    batch_data = [
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": "default_user",
            "role": "user",
            "message_content": "My favorite color is blue.",
            "insert_at": datetime.now(),
        },
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": "default_user",
            "role": "assistant",
            "message_content": "That's a lovely color! I'll remember that.",
            "insert_at": datetime.now(),
        },
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": "default_user",
            "role": "user",
            "message_content": "I live in a city with a famous bridge.",
            "insert_at": datetime.now(),
        },
        {
            "memory_id": uuid.uuid4().hex,
            "user_id": "default_user",
            "role": "assistant",
            "message_content": "Oh, that sounds like an interesting place!",
            "insert_at": datetime.now(),
        },
    ]
    memory.insert(batch_data)
    print(f"Inserted initial batch data for user 'default_user'.")

    print("\n--- Starting Simulated Conversation ---")

    simulated_conversation = [
        "What did I tell you about my favorite color?",
        "Where do I live?",
        "What are the two things I told you earlier?"
    ]

    current_user_id = "default_user"

    for user_input in simulated_conversation:
        print(f"\nYou: {user_input}")
        if not user_input:
            continue

        ai_response = chat_with_memories(current_user_message=user_input, user_id=current_user_id)
        print(f"AI: {ai_response}")

    print("\n--- Simulated Conversation Ended ---")

if __name__ == "__main__":
    main()
