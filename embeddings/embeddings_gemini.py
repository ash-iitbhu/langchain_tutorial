from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

embedding = GoogleGenerativeAIEmbeddings(model= "models/gemini-embedding-exp-03-07")

result = embedding.embed_query("delhi is capital of india", output_dimensionality=32)

print(result)
print("*****")
print(len(result))

documents = ["I am ashish", "this is a river", "I am learning langchain"]
result_2 = embedding.embed_documents(documents, output_dimensionality=32)
print(result_2)
print(len(result_2))
