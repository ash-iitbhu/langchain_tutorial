from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv(override=True)

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        task="text-generation",
                                        pipeline_kwargs=dict(temperature =0.5))
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of India?")
print(result.content)