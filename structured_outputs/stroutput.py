from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")


template1 = PromptTemplate(template = "Write a detailed report on the topic: {topic}",
                           input_variables= ["topic"])

template2 = PromptTemplate(template = "summarize in 5 lines the follwing text./n {text}",
                           input_variables= ["text"])


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence in Healthcare"})

print(result)