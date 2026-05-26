import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("📄 PDF Question Answering System using RAG")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:

    # Extract text from PDF
    pdf_reader = PdfReader(pdf)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    # User question
    question = st.text_input("Ask a question from the PDF")

    if question:

        # Retrieve relevant chunks
        docs = vector_store.similarity_search(question)

        # Combine retrieved text
        context = " ".join(
            [doc.page_content for doc in docs]
        )

        # Load GPT-2 model
        generator = pipeline(
            "text-generation",
            model="gpt2"
        )

        # Prompt
        prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate response
        response = generator(
            prompt,
            max_length=300,
            temperature=0.3,
            num_return_sequences=1
        )

        # Extract generated text
        answer = response[0]["generated_text"]

        # Display answer
        st.subheader("Answer")
        st.write(answer)