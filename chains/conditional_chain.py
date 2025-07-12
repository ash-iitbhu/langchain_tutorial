from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")

parser1 = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(template="provide sentiment for the following feedback: {feedback} /n {format_instructions}",
                         input_variables=["feedback"],
                         partial_variables={"format_instructions": parser2.get_format_instructions()})

prompt2 = PromptTemplate(template="provide an appropriate response for the positive sentiment feedback, it should be only 1 response according to the feedback: {feedback} ",
                         input_variables=["feedback"])

prompt3 = PromptTemplate(template="provide an appropriate response for the negative sentiment feedback, it should be only 1 response according to the feedback: {feedback} ",
                         input_variables=["feedback"])


sentiment_chain = prompt1 | model | parser2

response_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive",
        prompt2 | model | parser1),
    (lambda x: x.sentiment == "negative",
        prompt3 | model | parser1),

    RunnableLambda(lambda x: "unable to process feedback")

)

chain = sentiment_chain | response_chain | parser1

print(chain.invoke({"feedback": "I love the new features of this product but other things are very bad!"}))


print("*******")

chain.get_graph().print_ascii()