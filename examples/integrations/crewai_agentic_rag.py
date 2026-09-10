# pip install crewai openai pixelmemory

from pixelmemory import Memory
from pixelmemory.context import Document, Text
from pixelmemory.config import DocumentSplitterParams
from crewai import Crew, Agent, Task, Process, LLM
from crewai.tools import tool

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination",
]

context = [
    Text(id="doc_id", embed=False),
    Document(id="content", chunk_params=DocumentSplitterParams(separators="token_limit", limit=400, overlap=40)),
]

website_knowledge = Memory(
    context=context,
    namespace="crewai_rag_example",
    table_name="web_pages",
    if_exists="replace_force"
)


@tool
def website_knowledge_tool(search_query: str, limit: int = 5) -> str:
    """Search for relevant information in the website knowledge."""
    results = website_knowledge.search(search_query, on="content", limit=limit)
    context = "\n\n".join(
        f"Content: {r['text']}\nSimilarity: {r['score']:.4f}" for r in results
    )
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
entries = [website_knowledge.Entry(doc_id=url, content=url) for url in urls]
website_knowledge.add(*entries)

# Kickoff crew
question = "What is reward hacking and what are some examples? Provide sources."
result = crew.kickoff(inputs={"question": question})
