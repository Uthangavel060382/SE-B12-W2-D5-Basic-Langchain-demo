
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Step 1: Load environment variables from .env
load_dotenv()

# Step 2: Read the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

# Step 3: Initialize the AI model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key
)

# Step 4: Send a question to the AI model
response = llm.invoke(
    "Explain Generative AI in three simple sentences."
)

# Step 5: Print the generated response
print("\nAI Response:")
print(response.content)
