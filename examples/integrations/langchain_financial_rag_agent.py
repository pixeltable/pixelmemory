# uv pip install langchain-openai pixeltable httpx "unstructured[pdf]"
import pixeltable as pxt
from pixelmemory import Memory
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# --- 1. Setup: Create two PixelMemory instances ---
# One for the knowledge base (financial reports) and one for conversation history.

# Knowledge Base Memory
kb_mem = Memory(
    namespace="financial_kb",
    table_name="reports",
    schema={
        'doc_id': pxt.Required[pxt.String],
        'report': pxt.Document,
        'company': pxt.String,
        'report_type': pxt.String,
        'year': pxt.Int
    },
    primary_key='doc_id',
    columns_to_index=['report', 'company', 'report_type', 'year'],
    if_exists="replace_force"
)

# Conversation History Memory
chat_mem = Memory(
    namespace="financial_chat",
    table_name="history",
    schema={
        'session_id': pxt.Required[pxt.String],
        'messages': pxt.Json,
    },
    primary_key='session_id',
    if_exists="replace_force"
)

# --- 2. Populate the Knowledge Base ---
# We'll use a sample Zacks Nvidia financial report and a fictional AMD report.
REPORT_URL_NVDA = "https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/rag-demo/Zacks-Nvidia-Report.pdf"
kb_mem.insert([
    {
        'doc_id': 'zacks-nvidia-report',
        'report': REPORT_URL_NVDA,
        'company': 'Nvidia',
        'report_type': 'Equity Research',
        'year': 2024
    },
    {
        'doc_id': 'fictional-amd-report',
        'report': REPORT_URL_NVDA, # Using same PDF for demo purposes
        'company': 'AMD',
        'report_type': 'Earnings Call Transcript',
        'year': 2023
    }
])
print("📚 Knowledge base populated with financial reports.")

# --- 3. The RAG Agent Logic ---
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

def financial_chat(session_id: str, user_text: str, filters: dict = None) -> str:
    """
    Handles a turn in a financial RAG conversation, with optional metadata filtering.
    """
    print(f"\n--- New Turn (Session: {session_id}) ---")

    # a. Retrieve conversation history
    history_data = chat_mem.where(chat_mem.session_id == session_id).collect()
    messages = [SystemMessage("You are a helpful financial analyst. You will answer questions based on the provided financial reports.")]
    if history_data:
        messages.extend([AIMessage(m['content']) for m in history_data[0]['messages'] if m['type'] == 'ai'])

    # b. Augment with RAG: Search the knowledge base
    print(f"💬 User says: '{user_text}'")
    rag_context = ""
    if user_text:
        chunk_view = kb_mem.chunk_views['report']
        
        # Calculate similarity across all chunks
        sim = chunk_view.text.similarity(user_text)

        # Start with the base view
        query = chunk_view

        # Apply filters if provided
        if filters:
            print(f"🔍 Applying filters: {filters}")
            conditions = [getattr(query, key) == value for key, value in filters.items()]
            if conditions:
                final_filter = conditions[0]
                for i in range(1, len(conditions)):
                    final_filter = final_filter & conditions[i]
                query = query.where(final_filter)

        # Order by similarity and get the top results
        results = query.order_by(sim, asc=False).limit(3).select(chunk_view.text).collect()
        if results:
            rag_context = "\n\n--- Relevant Report Excerpts ---\n" + "\n".join([r['text'] for r in results])
            print("🧠 Found relevant info in the report(s).")
        else:
            print(" DNC Couldn't find any relevant info for the given query and filters.")


    # c. Prepare the prompt with text and RAG context
    prompt_content = [{"type": "text", "text": user_text + rag_context}]
    messages.append(HumanMessage(content=prompt_content))

    # d. Invoke the LLM
    ai_response = llm.invoke(messages)

    # e. Save updated history
    updated_messages = messages + [ai_response]
    chat_mem.insert([{
        'session_id': session_id,
        'messages': [m.dict() for m in updated_messages],
    }], on_conflict='update')
    print("💾 Saved conversation history.")

    return ai_response.content

# --- 4. Run a Sample Support Conversation ---
SESSION_ID = f"financial_session_{uuid.uuid4()}"

# Turn 1: User asks a general question (searches all documents)
response1 = financial_chat(SESSION_ID, "What is the general outlook for the semiconductor industry?")
print(f"\n🤖 AI Response: {response1}")

# Turn 2: User asks a specific question about Nvidia, filtering by company
response2 = financial_chat(
    SESSION_ID,
    "What are the key growth drivers for them?",
    filters={'company': 'Nvidia'}
)
print(f"\n🤖 AI Response: {response2}")

# Turn 3: User asks about a different type of report from 2023
response3 = financial_chat(
    SESSION_ID,
    "What was discussed in the earnings call?",
    filters={'report_type': 'Earnings Call Transcript', 'year': 2023}
)
print(f"\n🤖 AI Response: {response3}")
