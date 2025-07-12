from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatOpenAI(model = "gpt-4")

response = model.invoke("what is the capital of India?")

print(response)
print(response.content)
