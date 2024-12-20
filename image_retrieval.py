import streamlit as st
import numpy as np
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
import faiss
import re
import torch
from pdf2image import convert_from_path
import requests
import json

# Initialize PaddleOCR and Sentence Transformer
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to("cuda" if torch.cuda.is_available() else "cpu")

# Function to split document text into smaller chunks
def split_into_chunks(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to extract text using OCR from image-based PDF
def extract_text_using_ocr(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    ocr_text = ""
    for page_num, page_image in enumerate(pages):
        page_image_np = np.array(page_image)
        result = ocr.ocr(page_image_np)
        for line in result[0]:
            ocr_text += " ".join([word_info[1] for word_info in line]) + "\n"
    return ocr_text

# Function to extract text from text-based PDF
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    document_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        page_text = page.get_text()
        document_text += page_text
    return document_text

# Function for direct answer extraction using regular expressions
def extract_exact_answer(query, document_text):
    if "hall number" in query.lower():
        match = re.search(r"hall number\s*is\s*(\d+)", document_text, re.IGNORECASE)
        if match:
            return f"Hall Number: {match.group(1)}"
    if "date" in query.lower():
        match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", document_text)
        if match:
            return f"Date found: {match.group(0)}"
    if "schedule" in query.lower():
        match = re.search(r"(event schedule[:\s]*)([\s\S]+?)(\n\n|$)", document_text, re.IGNORECASE)
        if match:
            return f"Event Schedule:\n{match.group(2)}"
    return None

# Function to retrieve multiple relevant chunks using FAISS
def retrieve_relevant_chunks(query, document_text, top_k=3):
    document_chunks = split_into_chunks(document_text)
    document_embeddings = embedding_model.encode(document_chunks, convert_to_tensor=True).cpu().numpy()
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)
    D, I = index.search(query_embedding, k=top_k)
    return [document_chunks[i] for i in I[0]]

# Function to interact with the LLaMA model via API
def get_llama_response(prompt, model="llama3.1", stream=False):
    url = "https://llm.neorains.com/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            data = response.json()
            actual_response = data.get('response')
            return actual_response if actual_response else "No response key found in the API response."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Function to generate a context-aware response using LLaMA
def get_llama_response_with_context(query, context, model="llama3.1"):
    prompt = f"Answer the question based strictly on the following context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return get_llama_response(prompt, model)

# Main function for document retrieval with improved relevance
def auto_retrieval_from_document(document_path, user_query):
    try:
        document_text = extract_text_from_pdf(document_path)
        if not document_text:
            document_text = extract_text_using_ocr(document_path)
        exact_answer = extract_exact_answer(user_query, document_text)
        if exact_answer:
            return get_llama_response_with_context(user_query, exact_answer)
        relevant_chunks = retrieve_relevant_chunks(user_query, document_text)
        combined_context = "\n".join(relevant_chunks)
        if combined_context:
            return get_llama_response_with_context(user_query, combined_context)
        return get_llama_response(user_query)
    except Exception as e:
        return f"Error during document retrieval: {e}"

# Streamlit App
def main():
    st.title("PDF Question Answering System")
    st.write("Upload a PDF document and ask a question about its content.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    user_query = st.text_input("Enter your question")

    if st.button("Get Answer") and uploaded_file and user_query:
        with st.spinner("Processing your request..."):
            # Save uploaded file to a temporary location
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process the document and query
            response = auto_retrieval_from_document(temp_path, user_query)

            # Display the result
            st.success("Response Retrieved")
            st.write(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
