
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

class PersonInfo(BaseModel):
    name: str = Field(description="The name of the person")
    city: str = Field(description="The city where the person lives")
    gender: Optional[Literal["M","F"]] = Field(description="gender of the person")
    age: int = Field(gt=18, lt=100, description="The age of the person")
    salary: Optional[float] = Field(description="The salary of the person")

class PersonInfoOutput(BaseModel):
    person_info: List[PersonInfo] = Field(description="List of person information extracted from the text")


parser = PydanticOutputParser(pydantic_object=PersonInfoOutput)

template = PromptTemplate(template = "From the given input text, extract person's name, city, gender, age and salary: {text} /n {format_instructions}",
                           input_variables= ["text"],
                           partial_variables={"format_instructions": parser.get_format_instructions()})


chain = template | model | parser

result = chain.invoke({"text": "Alice lives in New York, she is 30 years old, earns $50000 and identifies herself as a Female. Bob lives in Los Angeles, he is 35 years old"})

print(result)