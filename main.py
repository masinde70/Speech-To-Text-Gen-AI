
from dotenv import load_dotenv
from loguru import logger
from typing import List
from pydantic import BaseModel, HttpUrl, ValidationError
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

class URLListModel(BaseModel):
    urls: List[HttpUrl]

load_dotenv()

try:
    url_list_model = URLListModel(urls=[
        "https://storage.googleapis.com/aai-web-samples/langchain_agents_webinar.opus",
        "https://storage.googleapis.com/aai-web-samples/langchain_document_qna_webinar.opus",
        "https://storage.googleapis.com/aai-web-samples/langchain_retrieval_webinar.opus"
    ])
    URLs = url_list_model.urls
except ValidationError as e:
    logger.error(f"Invalid URLs: {e}")
    exit(1)

def create_docs(urls_list: List[str]):
    docs = []
    for url in urls_list:
        logger.info(f"Transcribing {url}")
        docs.append(AssemblyAIAudioTranscriptLoader(file_path=url).load()[0])
    return docs

def make_embedder():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def make_qa_chain(db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
        return_source_documents=True
    )

if __name__ == "__main__":
    logger.info('Transcribing files ... (may take several minutes)')
    docs = create_docs(URLs)

    logger.info('Splitting documents')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    for text in texts:
        text.metadata = {"audio_url": text.metadata["audio_url"]}

    logger.info('Embedding texts...')
    hf = make_embedder()
    db = Chroma.from_documents(texts, hf)

    logger.success('Ready for questions. Enter `e` to exit.')
    qa_chain = make_qa_chain(db)
    while True:
        q = input('Enter your question: ')
        if q.strip().lower() == 'e':
            logger.info('Exiting...')
            break
        result = qa_chain({"query": q})
        print(f"Q: {result['query'].strip()}")
        print(f"A: {result['result'].strip()}\n")
        print("SOURCES:")
        for idx, elt in enumerate(result['source_documents']):
            print(f"    Source {idx}:")
            print(f"        Filepath: {elt.metadata['audio_url']}")
            print(f"        Contents: {elt.page_content}")
        print('\n')


**To use loguru, just install it:**
```bash
pip install loguru
```
