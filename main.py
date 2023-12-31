from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


load_dotenv()

URLs = [
    "https://storage.googleapis.com/aai-web-samples/langchain_agents_webinar.opus",
    "https://storage.googleapis.com/aai-web-samples/langchain_document_qna_webinar.opus",
    "https://storage.googleapis.com/aai-web-samples/langchain_retrieval_webinar.opus"
]


def create_docs(urls_list):
    l = []
    for url in urls_list:
        print(f'Transcribing {url}')
        l.append(AssemblyAIAudioTranscriptLoader(file_path=url).load()[0])
    return l

def make_embedder():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def make_qa_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
        return_source_documents=True
    )


print('Transcribing files ... (may take several minutes)')
docs = create_docs(URLs)

print('Splitting documents')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# modify metadata because some AssemblyAI returned metadata is not in a compatible form for the Chroma db
for text in texts:
    text.metadata = {"audio_url": text.metadata["audio_url"]}

# Make vector DB from split texts
print('Embedding texts...')
hf = make_embedder()
db = Chroma.from_documents(texts, hf)

# Create the chain and start the program
print('\nEnter `e` to exit')
qa_chain = make_qa_chain()
while True:
    q = input('enter your question: ')
    if q == 'e':
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
