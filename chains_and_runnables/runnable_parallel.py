from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt1 = PromptTemplate(template = "explain in detail the following topic: {topic}",
                         input_variables=["topic"])
prompt2 = PromptTemplate(template = "provide a summary of the following topic in less than 100 words: {topic}",
                         input_variables=["topic"])


parser = StrOutputParser()



explainer_chain = RunnableParallel({
    "explanation": prompt1 | model | parser,
    "summary": prompt2 | model | parser
})

output = explainer_chain.invoke({"topic": "Support Vector Machines"})
print(output)