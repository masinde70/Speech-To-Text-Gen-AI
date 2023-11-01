# Speech-To-Text-Gen-AI
Retrieval Augmented Generation on audio data with LangChain and Chroma database. 
Retrieval Augmented Generation (RAG) allows you to add relevant documents as context when querying LLMs.

RAG on audio data using LangChain and Chroma 

Retrieval Augmented Generation (RAG) is a method to augment the relevance and transparency of Large Language Model (LLM) responses. In this approach, the LLM query retrieves relevant documents from a database and passes these into the LLM as additional context. RAG therefore helps improve the relevancy of responses by including pertinent information in the context, and also improves transparency by letting the LLM reference and cite source documents.

## Stack Used 

- AssemblyAI
- LangChain
- Hugging Face
- **Chroma** 
    - Chroma is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs
- OpenAI

## How to use this Application
- pip install the requirements
- Put your API keys in dotenv so that you can hide your keys before exposing them to the public
- run main.py to the terminal then start asking the question
for example "What is RAG?"