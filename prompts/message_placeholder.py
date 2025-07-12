import ast
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_template = ChatPromptTemplate([("system", "You are a customer support agent"),
                                    MessagesPlaceholder(variable_name="chat_history"),
                                    ("human", "{query}")])

chat_history = []

with open("prompts/chat_history.txt", "r") as f:
    for line in f:
        role, message = ast.literal_eval(line.strip())
        if role.lower() == "human":
            chat_history.append(HumanMessage(content=message))
        elif role.lower() == "ai":
            chat_history.append(AIMessage(content=message))
        else:
            raise ValueError(f"Unknown role: {role}")

print(chat_history)

prompt = chat_template.invoke({"chat_history": chat_history, "query": "Where is my refund?"})

print(prompt)
