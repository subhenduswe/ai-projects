from langchain.document_loaders import PyPDFLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Path to the directory where the downloaded model and tokenizer files are stored
model_path = "./bart-large-cnn/"

# Load the model and tokenizer from the local directory
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# Create a summarization pipeline using the loaded model and tokenizer
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

import streamlit as st
import tempfile
import os

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()

        page_contents = [doc.page_content for doc in docs]

        summary = summarizer(page_contents, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary)

        # Delete the temporary file
        os.remove(temp_path)

    return summaries


# Streamlit App
st.title("Text Summarizer from PDF")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.write("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for PDF {i + 1}:")
            st.write(summary)
