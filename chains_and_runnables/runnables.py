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
prompt2 = PromptTemplate(template = "provide a summary of the following text in less than 100 words: {text}",
                         input_variables=["text"])


parser = StrOutputParser()



explainer_chain = RunnableSequence(prompt1, model, parser)


branch_chain = RunnableBranch(
    (lambda x: len(x.split())>100, prompt2 | model | parser),
    RunnablePassthrough()
    )


def word_count(text: str) -> int:
    return len(text.split())

word_count_chain = RunnableSequence(explainer_chain,branch_chain, RunnableLambda(word_count))

output = word_count_chain.invoke({"topic": "Support Vector Machines"})
print(output)