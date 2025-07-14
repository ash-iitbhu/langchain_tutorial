from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")
embedding = GoogleGenerativeAIEmbeddings(model= "models/gemini-embedding-exp-03-07")

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"}),

     Document(page_content=(
        """The Grand Canyon is a natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        many tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),
]

vectorstore = FAISS.from_documents(docs, embedding)

#base_retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_type='mmr',
#                                                                                       search_kwargs={"k": 5, "lambda_mult": 0.5}),
#                                                                                       llm=model)

base_retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 5, "lambda_mult": 0.1})
compressor = LLMChainExtractor.from_llm(model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "What is photosynthesis?"
result = compression_retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
