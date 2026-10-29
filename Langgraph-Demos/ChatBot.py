from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite"
)

def ChatBotNode(state: ChatBotState) -> ChatBotState:
    result = llm.invoke(state["messages"])
    return {'messages': [result]}

graph = StateGraph(ChatBotState)
graph.add_node('ChatBotNode', ChatBotNode)
graph.add_edge(START, 'ChatBotNode')
graph.add_edge('ChatBotNode', END)
workflow = graph.compile(checkpointer=InMemorySaver())

image = workflow.get_graph().draw_mermaid_png()
with open("chatbot_graph.png", mode="wb") as f:
    f.write(image)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = workflow.invoke({'messages': HumanMessage(content=user_input)}, config={"configurable": {"thread_id": "1"}})
    print(f"AI: {response['messages'][-1].text}")