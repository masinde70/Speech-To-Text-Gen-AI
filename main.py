

# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv

# Import the logger object from the loguru library for advanced logging
from loguru import logger

# Import List type for type hinting lists
from typing import List

# Import BaseModel, HttpUrl, and ValidationError from pydantic for data validation
from pydantic import BaseModel, HttpUrl, ValidationError

# Import RetrievalQA chain for question-answering from langchain
from langchain.chains import RetrievalQA

# Import ChatOpenAI for interacting with OpenAI's GPT models via langchain
from langchain.chat_models import ChatOpenAI

# Import document loader for transcribing audio files using AssemblyAI
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader

# Import HuggingFaceEmbeddings for embedding text via pretrained models
from langchain.embeddings import HuggingFaceEmbeddings

# Import text splitter to break large documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Chroma vectorstore for storing and querying embeddings
from langchain.vectorstores import Chroma

# Define a Pydantic model for validating a list of URLs
class URLListModel(BaseModel):
    urls: List[HttpUrl]  # Accepts only valid HTTP URLs

# Load environment variables from the .env file (if present)
load_dotenv()

# Validate a hardcoded list of URLs using the URLListModel
try:
    url_list_model = URLListModel(urls=[
        "https://storage.googleapis.com/aai-web-samples/langchain_agents_webinar.opus",
        "https://storage.googleapis.com/aai-web-samples/langchain_document_qna_webinar.opus",
        "https://storage.googleapis.com/aai-web-samples/langchain_retrieval_webinar.opus"
    ])
    URLs = url_list_model.urls  # Store the validated list of URLs
except ValidationError as e:
    logger.error(f"Invalid URLs: {e}")  # Log an error if any URL is invalid
    exit(1)  # Exit the script due to invalid configuration

# Function to create documents from a list of URLs by transcribing audio to text
def create_docs(urls_list: List[str]):
    docs = []  # Initialize an empty list to hold documents
    for url in urls_list:
        logger.info(f"Transcribing {url}")  # Log which file is being transcribed
        # Use AssemblyAIAudioTranscriptLoader to transcribe the audio and load as a document
        docs.append(AssemblyAIAudioTranscriptLoader(file_path=url).load()[0])
    return docs  # Return the list of transcribed documents

# Function to create a HuggingFace embedder for converting text to embeddings
def make_embedder():
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Specify embedding model
    model_kwargs = {'device': 'cpu'}  # Set device to CPU
    encode_kwargs = {'normalize_embeddings': False}  # Do not normalize embeddings
    # Return the HuggingFaceEmbeddings object with the given configuration
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Function to create a RetrievalQA question-answering chain using the vector database
def make_qa_chain(db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Load GPT-3.5-turbo with deterministic output
    # Create a RetrievalQA chain with the language model and retriever
    return RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
        return_source_documents=True  # Also return the source documents used
    )

# Main script entry point
if __name__ == "__main__":
    logger.info('Transcribing files ... (may take several minutes)')
    docs = create_docs(URLs)  # Transcribe audio files into documents

    logger.info('Splitting documents')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Initialize text splitter
    texts = text_splitter.split_documents(docs)  # Split documents into chunks

    # Ensure each text chunk's metadata contains the audio_url
    for text in texts:
        text.metadata = {"audio_url": text.metadata["audio_url"]}

    logger.info('Embedding texts...')
    hf = make_embedder()  # Create the embedding model
    db = Chroma.from_documents(texts, hf)  # Store embeddings in a Chroma vector database

    logger.success('Ready for questions. Enter `e` to exit.')
    qa_chain = make_qa_chain(db)  # Initialize the question-answering chain

    # Interactive loop to accept user questions
    while True:
        q = input('Enter your question: ')  # Prompt user for a question
        if q.strip().lower() == 'e':  # Exit if user enters 'e'
            logger.info('Exiting...')
            break
        result = qa_chain({"query": q})  # Run the question through the QA chain
        print(f"Q: {result['query'].strip()}")  # Print the question
        print(f"A: {result['result'].strip()}\n")  # Print the answer
        print("SOURCES:")  # Print the sources used for answering
        for idx, elt in enumerate(result['source_documents']):
            print(f"    Source {idx}:")
            print(f"        Filepath: {elt.metadata['audio_url']}")
            print(f"        Contents: {elt.page_content}")
        print('\n')

# The lines below are not part of the script, but instructions on installing loguru for logging
# **To use loguru, just install it:**
# ```bash
# pip install loguru
# ```


