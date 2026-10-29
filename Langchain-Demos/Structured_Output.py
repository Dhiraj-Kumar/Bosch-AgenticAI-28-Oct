from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

class Feedback(BaseModel):
    name: str = Field(description="name of the participant")
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the overall feedback")
    highlights: list[str] = Field(description="List of highlights of the training program")
    lowlights: list[str] = Field(description="List of lowlights of the training program")
    rating: str = Field(description="Rating of the overall training program")

structured_model = model.with_structured_output(Feedback)

response = structured_model.invoke("The Java Fullstack training program was well-structured and covered essential modules like Core Java, Spring Boot, Hibernate, and Angular. The hands-on projects and live coding sessions made it easier to apply concepts in real-world scenarios. The trainer was knowledgeable and supportive, and the sessions on Git and deployment provided a complete view of end-to-end development. However, the pace during the Spring Boot section felt a bit fast, and more time for practice would have been helpful. Additionally, a dedicated session on debugging and code optimization could enhance the learning experience. Some front-end sessions, especially on Angular, felt rushed, and could benefit from more real-time examples. Out of 5 I would give 4 rating for this program. Feedback given by Dhiraj Kumar. Email address - Dhiraj.Kumar@gmail.com")

print(response.sentiment)
print(response.name)