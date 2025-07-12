from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"truncate_dim":32})

result = embedding.embed_query("I am ashish")

print(result)
print(len(result))

print("******")

documents = ["I am ashish", "this is a river", "I am learning langchain"]
result_2 = embedding.embed_documents(documents)
print(result_2)
print(len(result_2))
