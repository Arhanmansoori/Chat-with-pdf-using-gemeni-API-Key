from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain_google_genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import itertools

# Directly declared API keys
API_KEYS = [
    "your api key ",
]

# Create a cycle iterator for the API keys
api_key_cycle = itertools.cycle(API_KEYS)

# Function to configure API with a specific key
def configure_api():
    api_key = next(api_key_cycle)
    genai.configure(api_key=api_key)
    return api_key

# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return splitter.split_text(text)

# Function to generate vector store for text chunks
def get_vector_store(chunks):
    api_key = configure_api()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain using Google Gemini model
def get_conversational_chain():
    api_key = configure_api()
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Function to handle user queries and provide responses
def user_input(user_question):
    api_key = configure_api()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

# Main function to process PDF and ask questions
def ask_from_pdf(pdf_path):
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    print("PDF processed. You can now ask questions.")
    while True:
        user_question = input("Enter your question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        response = user_input(user_question)
        print(f"Answer: {response['output_text']}")

# Example usage
pdf_path = "ARHAN OCTOBER RESUME with photo.pdf"  # Replace with your actual PDF file path
ask_from_pdf(pdf_path)
