import streamlit as st
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

# Ensure required NLTK data is downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")  # If using Lemmatizer in the future
nltk.download("omw-1.4")  # Optional wordnet extensions
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

def create_wordcloud(text, max_words=500):
    wordcloud = WordCloud(width=800, height=400, background_color="white", max_words=max_words).generate(text)
    return wordcloud

def main():
    st.title("PDF to WordCloud Generator")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        with st.spinner("Processing text..."):
            tokens = preprocess_text(extracted_text)
            processed_text = " ".join(tokens)

        with st.spinner("Generating WordCloud..."):
            wordcloud = create_wordcloud(processed_text)

        st.success("WordCloud generated successfully!")

        # Display the WordCloud
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
