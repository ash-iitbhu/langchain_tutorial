from langchain_community.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

retriever = WikipediaRetriever(top_k_results=5, language="en")


class Movie(BaseModel):
    movie: str = Field(description="movie name")
    release_date: str = Field(description="movie release date in string format")

class ResultSchema(BaseModel):
    movies: list[Movie] = Field(description="list of movies with their release dates")

parser1 = PydanticOutputParser(pydantic_object=ResultSchema)

prompt1 = PromptTemplate(template ="extract the name of the movies from the given context:{context} /n {format_instructions}",
                        input_variables=["context"],
                        partial_variables={"format_instructions": parser1.get_format_instructions()})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = retriever | RunnableLambda(format_docs) | RunnableLambda(lambda x: {"context": x})| prompt1 | model | parser1

result = chain.invoke("Akshay Kumar")
print(result)