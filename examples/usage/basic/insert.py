import uuid
from datetime import datetime

import pixeltable as pxt
from pixelmemory import Memory

schema = {
    "memory_id": pxt.Required[pxt.String],
    "role": pxt.String,
    "content": pxt.String,
    "insert_at": pxt.Timestamp,
}

memory = Memory(
    namespace="chatbot",
    table_name="chat_history",
    schema=schema,
    columns_to_index=["content"],
    embedding_model="intfloat/e5-large-v2",
    if_exists="replace_force",
)

memory_id = uuid.uuid4().hex
data = [
    {
        "memory_id": memory_id,
        "role": "user",
        "content": "Hello, how are you?",
        "insert_at": datetime.now(),
    },
    {
        "memory_id": memory_id,
        "role": "assistant",
        "content": "I'm doing well, thank you! How about you?",
        "insert_at": datetime.now(),
    },
]

memory.insert(data)

# return all rows
res = memory.collect()

for row in res:
    print(f"{row['role']}: {row['content']}")