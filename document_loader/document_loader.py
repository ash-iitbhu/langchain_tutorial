import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

pdf_path = "data/Ashish Agarwal - Resume 2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")


loader = PyPDFLoader(pdf_path)
documents = loader.load()

resume = ""
for document in documents:
    resume += document.page_content


class Resume(BaseModel):
    name: str = Field(description="Name of the person")
    email: str = Field(description="Email address of the person")
    phone: Optional[str] = Field(default=None, description="Phone number of the person")
    skills: List[str] = Field(description="List of skills")
    education: List[str] = Field(description="List of educational qualifications")
    experience: List[str] = Field(description="List of work experiences")

parser = PydanticOutputParser(pydantic_object =Resume)

prompt = PromptTemplate(
    template="Extract the following information from the resume: {resume}\n{format_instructions}",
    input_variables=["resume"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt|model|parser

result = chain.invoke({"resume": resume})

print(result)