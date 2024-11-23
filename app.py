import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from summa.summarizer import summarize

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function for BERT summarization
def bert_summarize(text, max_length=130):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']

# Function for TextRank summarization
def textrank_summarize(text, ratio=0.2):
    return summarize(text, ratio=ratio)

# Streamlit interface
st.title("Extractive Text Summarization Tool for Novels")
st.write("Upload a novel (PDF format) to get extractive summaries using two methods: **BERT** and **TextRank**.")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    # Extract text from PDF
    st.info("Extracting text from the uploaded PDF...")
    text = extract_text_from_pdf(uploaded_file)
    
    if not text.strip():
        st.error("The uploaded file does not contain extractable text!")
    else:
        st.success("Text extracted successfully!")

        # Summarization controls
        st.write("### Summarization Options")
        max_length = st.slider("Maximum length of BERT summary", 50, 300, 130)
        textrank_ratio = st.slider("TextRank summarization ratio", 0.1, 0.5, 0.2, step=0.05)

        # Generate summaries
        if st.button("Generate Summaries"):
            st.info("Generating BERT Summary...")
            bert_summary = bert_summarize(text[:1024], max_length=max_length)  # Limit input for BERT

            st.info("Generating TextRank Summary...")
            textrank_summary = textrank_summarize(text, ratio=textrank_ratio)

            # Display results
            st.write("## Summarization Results")
            st.write("### BERT Summary")
            st.success(bert_summary)
            st.write("### TextRank Summary")
            st.info(textrank_summary)

        st.write("You can choose which summarization method suits your objective!")

# Footer
st.write("---")
st.write("Developed using **Streamlit**, **Transformers**, and **Summa**.")
