from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

parser = JsonOutputParser()
template = PromptTemplate(template = "From the given input text, extract person's name, city names and amount: {text} /n {format_instructions}",
                           input_variables= ["text"],
                           partial_variables={"format_instructions": parser.get_format_instructions()})


chain = template | model | parser

result = chain.invoke({"text": "Alice lives in New York and Bob lives in Los Angeles. They spent $100 and $200 respectively."})

print(result)