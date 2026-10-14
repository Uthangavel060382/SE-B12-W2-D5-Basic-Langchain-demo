
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing")

# 2. Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1-mini"
)

# 3. Define the prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly AI teacher. "
        "Explain concepts in simple English "
        "and provide a practical example."
    ),
    (
        "human",
        "{question}"
    )
])

# 4. Create the output parser
parser = StrOutputParser()

# 5. Build the LangChain chain
chain = prompt | llm | parser

# 6. Accept and process questions
print("Welcome to the GenAI Learning Assistant!")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        answer = chain.invoke({
            "question": question
        })

        print("\nAI:", answer, "\n")

    except Exception as error:
        print("Error:", error)
