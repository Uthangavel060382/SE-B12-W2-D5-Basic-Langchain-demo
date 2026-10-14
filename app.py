
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# STEP 1: Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# STEP 2: Configure Streamlit page
st.set_page_config(
    page_title="My First GenAI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My First GenAI Chatbot")
st.caption("Powered by LangChain + OpenAI + Streamlit")


# STEP 3: Validate API key
if not api_key:
    st.error("OPENAI_API_KEY is missing from .env")
    st.stop()


# STEP 4: Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.3
)


# STEP 5: Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a friendly AI tutor. "
        "Explain concepts in simple English "
        "and give practical examples."
    ),
    ("placeholder", "{history}"),
    ("human", "{question}")
])


# STEP 6: Create LangChain chain
parser = StrOutputParser()

chain = prompt | llm | parser


# STEP 7: Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# STEP 8: Add clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()


# STEP 9: Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# STEP 10: Accept user question
question = st.chat_input("Ask me anything...")


# STEP 11: Generate answer
if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Convert stored messages to LangChain history
    history = []

    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            history.append(
                AIMessage(content=message["content"])
            )

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke({
                    "history": history,
                    "question": question
                })

                st.markdown(answer)

            except Exception as error:
                st.error(f"Error: {error}")
                st.stop()

    # Save messages after successful response
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
