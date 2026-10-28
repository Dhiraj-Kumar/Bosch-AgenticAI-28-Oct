from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

conversation_history = []

while True:
    user_input = input("You: ")
    conversation_history.append(user_input)
    if user_input.lower() == "exit":
        break
    response = model.invoke(conversation_history)
    conversation_history.append(response.text)
    print(f"AI: {response.text}")

print(conversation_history)