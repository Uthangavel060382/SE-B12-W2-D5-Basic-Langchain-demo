
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Step 1: Load environment variables
load_dotenv()

# Step 2: Validate API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in .env")

# Step 3: Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key
)

# Step 4: Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor."),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

# Step 5: Create output parser
parser = StrOutputParser()

# Step 6: Create LangChain chain
chain = prompt | llm | parser

# Step 7: Initialize conversation history
history = []

print("Welcome to LangChain AI Tutor!")
print("Type 'exit' to stop.\n")

# Step 8: Start chatbot
while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        answer = chain.invoke({
            "history": history,
            "question": question
        })

        print("\nAI:", answer, "\n")

        # Save conversation history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

    except Exception as error:
        print("Error:", error)
