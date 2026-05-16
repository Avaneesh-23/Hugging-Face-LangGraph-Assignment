from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Tavily Search
tavily = TavilyClient()


# State schema
class AgentState(TypedDict):
    question: str
    search_result: str
    final_answer: str


# Node 1: Search Web
def search_web(state: AgentState):
    query = state["question"]

    response = tavily.search(
        query=query,
        max_results=3
    )

    results = []

    for r in response["results"]:
        results.append(r["content"])

    return {
        "search_result": "\n".join(results)
    }


# Node 2: Generate Final Answer
def generate_answer(state: AgentState):
    question = state["question"]
    search_result = state["search_result"]

    prompt = f"""
You are a helpful AI research assistant.

Answer the question based on the search results.

Question:
{question}

Search Results:
{search_result}
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }


# Build LangGraph Workflow
graph = StateGraph(AgentState)

graph.add_node("search_web", search_web)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("search_web")

graph.add_edge("search_web", "generate_answer")
graph.add_edge("generate_answer", END)

# Compile Graph
app = graph.compile()


# Run Application
question = input("Enter your question: ")

result = app.invoke({
    "question": question
})

print("\nFinal Answer:\n")
print(result["final_answer"])