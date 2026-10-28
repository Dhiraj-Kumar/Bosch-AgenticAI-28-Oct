from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from datetime import date
import requests
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

@tool
def getWeatherData(city: str):
    """
    Get the current weather based on the city name from openweathermap API
    """
    response = requests.get("https://api.openweathermap.org/data/2.5/weather", {
        "q": city,
        "appid": os.getenv("WEATHER_API_KEY"),
        "units": "metric"
    })
    if response.status_code!=200:
        return "Unable to fetch data from API"
    return response.json()


agent = create_agent(
    model=model,
    system_prompt=f"You are a helpful AI weather agent who answers queries about the latest weather information as of today {date.today()} with a bit of humour. Keep your tone professional while answering the questions. Use emojis while generating response to make your responses looks good. Do not make up things and give only real time information. Always answer with updated information using tools. If you cannot access tools or does not have latest information, simply refuse politely. You only answer queries related to the weather updates. Do not answer any other questions and refuse politely",
    checkpointer=InMemorySaver(),
    tools=[getWeatherData]
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke({
    "messages": {
        "role": "user", "content": user_input
        }
    }, config={"configurable": {"thread_id": "1"}})
    print(f"AI: {response["messages"][-1].text}")