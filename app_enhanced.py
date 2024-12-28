import os
import uuid
import torch
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("Conversational RAG with PDF Uploads and Chat History")

# Sidebar for API key and PDF upload
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter your Groq API key:", type="password")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Truncate chat history
def truncate_history(chat_history, max_messages=5):
    return chat_history[-max_messages:]

# Process uploaded PDFs
def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_filename = f"./temp_{uuid.uuid4().hex}.pdf"
        with open(temp_filename, "wb") as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filename)
        documents.extend(loader.load())
        os.remove(temp_filename)  # Clean up temporary file
    return documents

# Main chat function
def chat():
    if not api_key:
        st.warning("Please enter the Groq API key.")
        return

    if uploaded_files:
        documents = process_uploaded_files(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        # vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()


        """contextualize_q_prompt: This template focuses on reformulating user questions into standalone questions by leveraging the chat history.
            Here's a breakdown of what it does:
            Purpose
            - Reformulate Context-Dependent Questions:
                    In conversations, users often ask follow-up questions or refer to earlier context implicitly. For example:
                    User: "What is its price?"
                    Without context, it's unclear what "its" refers to. The reformulation process transforms it into a standalone question like:
                    "What is the price of the product we discussed earlier?"
            - Improve Retrievability:
                    Standalone questions are easier for the retriever to match with relevant documents or content, as they contain all necessary context."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Reformulate the question to make it standalone."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It"),
            retriever,
            contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a knowledgeable assistant. Answer the question using the provided context."),
                ("human", "Context: {context}\n\nQuestion: {input}")
            ]   
        )

        question_answer_chain = create_stuff_documents_chain(
            ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It"), qa_prompt
        )
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session):
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask any question to chat with the PDF contents:")
        if user_input:
            session_history = get_session_history(session_id)
            session_history.messages = truncate_history(session_history.messages, max_messages=10)  # Token Management: Adjusting prompts and retaining only the last 10 chat messages.

            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write("AI Assistant:", response["answer"])
                st.write("Chat History:", session_history.messages)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Call the chat function
if api_key:
    chat()
else:
    st.warning("Please enter the Groq API key and click enter, then upload your document(s).")
