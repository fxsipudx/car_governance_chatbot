from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load your .env with OpenAI key
load_dotenv()

# Reconnect to ChromaDB with the same persist directory
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# Ask a governance-related question
query = "What is ASIL D?"
results = vectorstore.similarity_search(query, k=3)

# Show top 3 matched text chunks
for i, doc in enumerate(results):
    print(f"\n🔹 Match {i+1}:\n{doc.page_content[:500]}")