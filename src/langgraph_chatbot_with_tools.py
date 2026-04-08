# LangGraph Chatbot with Tools
# This notebook demonstrates how to build a chatbot using LangGraph that can utilize tools like Wikipedia and Arxiv to retrieve information and answer user queries. We will use Groq LLM for language understanding and
import os
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph.message import add_messages
## Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

## Arxiv And Wikipedia tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

wiki_tool.invoke("who is Sharukh Khan")

arxiv_tool.invoke("Attention is all you need")

tools=[wiki_tool]

## Langgraph Application
class State(TypedDict):
  messages:Annotated[list,add_messages]

load_dotenv()
# Set your Groq API Key
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile",temperature=0)
llm_with_tools=llm.bind_tools(tools=tools)

def chatbot(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

# Create the graph builder
graph_builder= StateGraph(State)
graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START,"chatbot")

graph=graph_builder.compile()

config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10}

# Test the chatbot with tools
user_input="Who is Shahrukh Khan?"
events=graph.stream(
     {"messages": [("user", user_input)]},stream_mode="values", config=config
)
for event in events:
    try:
        event["messages"][-1].pretty_print()
    except KeyError:
        pass

# Test the chatbot with a question that requires retrieval
user_input = "Tell me something about the politics of India"
events = graph.stream(
    {"messages": [("user", user_input)]},stream_mode="values", config=config
)
for event in events:
    try:
        event["messages"][-1].pretty_print()
    except KeyError:
        pass