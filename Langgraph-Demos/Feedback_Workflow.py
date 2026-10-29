from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
import os
import resend

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

class Feedback(BaseModel):
    participant_name: str = Field(description="name of the participant")
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the overall feedback")
    highlights: list[str] = Field(description="List of highlights of the training program")
    lowlights: list[str] = Field(description="List of lowlights of the training program")
    rating: str = Field(description="Rating of the overall training program")
    email_address: str = Field(description="Email address of the participant")


class AgentState(MessagesState):
    feedback: str
    participant_name: str
    sentiment: str
    rating: str
    email: str
    email_address: str

def AnalyzeFeedback(state: AgentState) -> AgentState:
    structured_model = model.with_structured_output(Feedback)
    response = structured_model.invoke(state['feedback'])
    return {
        "sentiment": response.sentiment,
        "participant_name": response.participant_name,
        "rating": response.rating,
        "email_address": response.email_address
    }

def GeneratePositiveEmail(state: AgentState) -> AgentState:
    response = model.invoke(f"Write a thank you email to the participant for giving positive feedback about the training program. Email is from 'Training Team'. Use HTML format and make it elegant and professional. Participant name: {state['participant_name']}.")
    print(response.text)
    return {"email": response.text}

def GenerateNegativeEmail(state: AgentState) -> AgentState:
    response = model.invoke(f"Write an apology email to the participant for giving negative feedback about the training program. Email is from 'Training Team'. Use HTML format and make it elegant and professional. Participant name: {state['participant_name']}.")
    print(response.text)
    return {"email": response.text}

def CheckCondition(state: AgentState) -> Literal["GeneratePositiveEmail", "GenerateNegativeEmail"]:
    return "GeneratePositiveEmail" if state['sentiment']=="positive" else "GenerateNegativeEmail"

@tool
def SendEmailTool(email_address: str, body: str) -> str:
    """
    Tool for sending email using Resend API.
    """
    decision = interrupt("Do you want to send this email?")
    if decision.lower() == "yes":    
        resend.api_key=os.getenv("RESEND_API_KEY")
        params: resend.Emails.SendParams = {
            "from": "training@resend.dev",
            "to": email_address,
            "subject": "Reply from Training Manager",
            "html": body
        }
        result = resend.Emails.send(params)
    

mytools = [SendEmailTool]
model_with_tools = model.bind_tools(mytools)
tool_node = ToolNode(mytools)

def SendEmail(state:AgentState) -> AgentState:
    if not state['messages']:
        instruction = HumanMessage(content=f"User the SendEmailTool to send email. email_address: {state['email_address']}; body: {state['email']}")
        response = model_with_tools.invoke([instruction])
        return {"messages": [instruction, response]}
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}

graph = StateGraph(AgentState)
graph.add_node("AnalyzeFeedback", AnalyzeFeedback)
graph.add_node("GeneratePositiveEmail", GeneratePositiveEmail)
graph.add_node("GenerateNegativeEmail", GenerateNegativeEmail)
graph.add_node("SendEmail", SendEmail)
graph.add_node("tools", tool_node)

graph.add_edge(START, "AnalyzeFeedback")
graph.add_conditional_edges("AnalyzeFeedback", CheckCondition)
graph.add_edge("GeneratePositiveEmail", "SendEmail")
graph.add_edge("GenerateNegativeEmail", "SendEmail")
graph.add_conditional_edges("SendEmail", tools_condition)
graph.add_edge("tools", "SendEmail")

workflow = graph.compile(checkpointer=InMemorySaver())

image = workflow.get_graph().draw_mermaid_png()
with open("FeedbackAnalyzer.png", mode="wb") as f:
    f.write(image)

# workflow.invoke(
#     {
#         "feedback": "The Java Fullstack training program did not meet expectations. Although it covered key technologies like Core Java, Spring Boot, Hibernate, and Angular, several sections were rushed—especially Spring Boot and Angular—leaving little time for hands-on practice. The lack of dedicated sessions on debugging and code optimization further reduced the program’s effectiveness. While the trainer was knowledgeable, the pace and structure limited meaningful learning and real-world application. Overall, the program requires better pacing and more practical focus. I would rate it 2 out of 5. Feedback given by Dhiraj Kumar. Email address - dhiraj2001@gmail.com"
#     }, config={"configurable": {"thread_id": "1"}}
# )

response = workflow.invoke(
    {
        "feedback": "The Java Fullstack training program was well-structured and covered essential modules like Core Java, Spring Boot, Hibernate, and Angular. The hands-on projects and live coding sessions made it easier to apply concepts in real-world scenarios. The trainer was knowledgeable and supportive, and the sessions on Git and deployment provided a complete view of end-to-end development. However, the pace during the Spring Boot section felt a bit fast, and more time for practice would have been helpful. Additionally, a dedicated session on debugging and code optimization could enhance the learning experience. Some front-end sessions, especially on Angular, felt rushed, and could benefit from more real-time examples. Out of 5 I would give 4 rating for this program. Feedback given by Dhiraj Kumar. Email address - dhiraj2001@gmail.com"
    }, config={"configurable": {"thread_id": "1"}}
)

interrupt_response = response.get("__interrupt__")
if interrupt_response:
    print("Workflow interrupted")
    user_input = input("Your Response: ").lower()
    result = workflow.invoke(
        Command(resume=user_input),
        config={"configurable": {"thread_id": "1"}}
    )