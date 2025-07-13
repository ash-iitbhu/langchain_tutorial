from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv(override=True)

model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        task="text-generation",
                                        pipeline_kwargs=dict(temperature =0.5))
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template="Prepare short and crisp notes from the follwing text: {text}",
                         input_variables=["text"])

prompt2 = PromptTemplate(template="Prepare 5 question answer pairs from the following: {text}",
                         input_variables=["text"])

prompt3 = PromptTemplate(template="create a single document containting notes --> {notes} and question answer pairs --> {qapairs}",
                         input_variables=["notes", "qapairs"])

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "qapairs": prompt2 | model1 | parser
})

final_chain = parallel_chain | prompt3 | model2 | parser

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

output = final_chain.invoke({"text":text})

print(output)

print("*****")

final_chain.get_graph().print_ascii()

