#dynamic input

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key
)

print("Welcome to My First GenAI Application!")

# Get a question from the user
question = input("Enter your question: ")

# Call the AI model
response = llm.invoke(question)

# Display the response
print("\nAI Answer:")
print(response.content)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI tutor. Explain every concept "
        "in simple language with one practical example."
    ),
    (
        "human",
        "{question}"
    )
])