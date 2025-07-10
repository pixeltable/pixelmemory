# pip install crewai langchain-openai pixelmemory

import pixeltable as pxt
from pixelmemory import Memory
from crewai import Crew, Agent, Task, Process, LLM
from crewai.tools import tool

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination",
]

website_knowledge = Memory(
    namespace="crewai_rag_example",
    table_name="web_pages",
    schema={'doc_id': pxt.Required[pxt.String], 'content': pxt.Document},
    primary_key='doc_id',
    columns_to_index=['content'],
    document_iterator_kwargs={'separators': 'token_limit', 'limit': 400, 'overlap': 40},
    if_exists="replace_force"
)

@tool
def website_knowledge_tool(search_query: str, limit: int = 5) -> str:
    """Search for relevant information in the website knowledge."""
    chunk_view = website_knowledge.chunk_views['content']
    sim = chunk_view.text.similarity(search_query)
    results = chunk_view.order_by(sim, asc=False).limit(limit).select(chunk_view.text, chunk_view.doc_id, similarity=sim).collect()
    context = "\n\n".join([
        f"Source: {r['doc_id']}\nContent: {r['text']}\nSimilarity: {r['similarity']:.4f}"
        for r in results
    ])
    return context

researcher = Agent(
    role="AI Research Analyst",
    goal="Answer questions accurately, using the website knowledge tool to find relevant information.",
    backstory="You are a meticulous analyst who uses tools to find information and follows instructions precisely, never providing information without citing your sources.",
    verbose=True,
    allow_delegation=False,
    llm=LLM(model="gpt-4o-mini", temperature=0),
    tools=[website_knowledge_tool],
)

task = Task(
    description=(
        "Answer the following question: '{question}'\n\n"
        "Use the `website_knowledge_tool` to find the necessary information. "
        "For each piece of information in your answer, you must cite the source URL."
    ),
    expected_output="A comprehensive answer to the question, with each statement supported by a citation from the website.",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
)

# Add knowledge to memory
data_to_insert = [{'doc_id': url, 'content': url} for url in urls]
website_knowledge.insert(data_to_insert)

# Kickoff crew
question = "What is reward hacking and what are some examples? Provide sources."
result = crew.kickoff(inputs={"question": question})
