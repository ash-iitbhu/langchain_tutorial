from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

messages = [
    SystemMessage(content="You are an experienced doctor."),
    HumanMessage(content="What are sympoms of diabetes?")
    ]

response = model.invoke(messages)
messages.append(AIMessage(content=response.content))

print(messages)
                 