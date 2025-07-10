# uv pip install langchain-openai pixeltable httpx
import pixeltable as pxt
from pixelmemory import Memory
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 1. Set up PixelMemory to store image references
mem = Memory(
    namespace="langchain_multimodal",
    table_name="image_store",
    schema={'image_id': pxt.Required[pxt.String], 'image_url': pxt.String},
    primary_key='image_id',
    if_exists="replace_force"
)

# 2. Store an image URL in PixelMemory
IMAGE_ID = "boardwalk_image"
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
mem.insert([{'image_id': IMAGE_ID, 'image_url': IMAGE_URL}])
print(f"Stored image '{IMAGE_ID}' in PixelMemory.")

# 3. Retrieve the image URL from PixelMemory
retrieved_data = mem.where(mem.image_id == IMAGE_ID).select(mem.image_url).collect()
retrieved_url = retrieved_data[0]['image_url']
print(f"Retrieved image URL: {retrieved_url}")

# 4. Use the retrieved URL with a multimodal LangChain model
print("\nAsking model to describe the image...")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the weather in this image:"},
        {"type": "image_url", "image_url": {"url": retrieved_url}},
    ]
)

response = llm.invoke([message])
print(f"\nAI Response:\n{response.content}")
