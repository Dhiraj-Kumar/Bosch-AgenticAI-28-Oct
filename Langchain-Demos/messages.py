from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

conversation_history = [
    SystemMessage(content="""
    You are a helpful AI assistant who answers queries of the user with a bit of humour. Keep your tone professional while answering the questions. Use emojis while generating response to make your responses looks good.
    """)
]

while True:
    user_input = input("You: ")
    conversation_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(conversation_history)
    conversation_history.append(AIMessage(content=response.text))
    print(f"AI: {response.text}")

print(conversation_history)