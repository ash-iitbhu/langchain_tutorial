from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

template = ChatPromptTemplate([("system", "You are a {domain} expert"),
                               ("human", "explain in simple terms, what is {topic}")],
                               input_variables=["domain", "topic"])

chain = template | model

response = chain.invoke({"domain": "AI", "topic": "transformer architecture"})
print(response.content)