from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"truncate_dim":256})

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "best cricketer of all time"

document_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

similarity = cosine_similarity([query_embeddings],document_embeddings)

index, score = sorted(list(enumerate(similarity[0])),key = lambda x: x[1])[-1]

print(documents[index])
