# Car Governance Chatbot (RAG-based)

This project is a prototype Retrieval-Augmented Generation (RAG) chatbot tailored for automotive governance use-cases. It uses LangChain, OpenAI LLMs, and a Chroma vector store to answer domain-specific questions based on internal PDF documents.

## 🔧 Problem It Solves

Automotive companies often face the challenge of hallucinations when using LLMs for regulatory or governance-related queries. This project aims to solve that by grounding answers in actual documentation using Retrieval-Augmented Generation (RAG).

## 📚 What It Does

- Ingests and processes internal automotive PDF documents.
- Splits them into overlapping chunks for efficient vector storage.
- Embeds and stores them in a Chroma vector database.
- Queries are matched against this database and fed into an OpenAI LLM with context to generate reliable responses.
- A Streamlit-based frontend is provided for easy interaction.
- **NEW**: Chain-of-Verification (CoVe) implementation for enhanced answer accuracy.

## 🚀 Stack

- **LLM**: OpenAI GPT
- **RAG Framework**: LangChain
- **Embeddings**: OpenAI Embeddings
- **Vector DB**: Chroma
- **Frontend**: Streamlit
- **Verification**: Chain-of-Verification (CoVe) for reduced hallucinations

## 🧠 How It Works

1. **load_docs.py** – Loads and parses all PDFs from the `docs/` folder.
2. **split_docs.py** – Splits the content into chunks and stores them as `.pkl`.
3. **embed_store.py** – Embeds chunks and stores/retrieves them in/from Chroma.
4. **rag_chatbot.py** – CLI chatbot that pulls relevant documents and answers using OpenAI + LangChain.
5. **streamlit_app.py** – Web UI for chatting with the bot.
6. **test_query.py** – Utility for testing vector retrieval performance.

## 🔍 Chain-of-Verification (CoVe)

We've implemented an enhanced version with Chain-of-Verification to further reduce hallucinations:

- **Standard Version**: `rag_chatbot.py` (original implementation)
- **CoVe Version**: `rag_chatbot_COVE_version.py` (with verification)

### How CoVe Works:
1. Generates initial answer using RAG
2. Creates verification questions about the answer
3. Retrieves additional context for verification
4. Revises the answer based on verification evidence

### Using the CoVe Version:
To activate Chain-of-Verification:
```bash
# Option 1: Use directly
python rag_chatbot_COVE_version.py "Your question here"

# Option 2: Replace the standard version
rm rag_chatbot.py
mv rag_chatbot_COVE_version.py rag_chatbot.py
```

The CoVe version is fully backward compatible and includes a `--no-verification` flag to disable verification if needed.

## 📁 Folder Structure

```
car_governance_chatbot/
│
├── docs/                  # Raw automotive PDFs
├── data/                  # Pickled chunked data
├── chroma_store/          # Vector DB files
├── load_docs.py
├── split_docs.py
├── embed_store.py
├── rag_chatbot.py              # Standard RAG version
├── rag_chatbot_COVE_version.py # Enhanced CoVe version
├── streamlit_app.py
├── test_query.py
└── .env
```

## 🗒️ Notes

- This is a prototype with a small PDF base.
- Can be scaled with a larger corpus and fine-tuned prompts.
- CoVe version provides better accuracy at the cost of slightly increased response time.

## ✅ Status

Functional prototype with both standard RAG and Chain-of-Verification implementations. Future improvements may include document tagging, confidence scores, and feedback loops.

## 🧑‍💻 Author

Shubham Jena – Based on an idea from a friend working in automotive compliance.

---

Feel free to contribute or adapt for other industries facing similar LLM hallucination issues.