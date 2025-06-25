
import datetime as dt
import streamlit as st
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from embed_store import load_vectorstore
from rag_chatbot import Settings, AUTOMOTIVE_PROMPT

# Page config & theme 
st.set_page_config(page_title="Car Governance Chatbot", page_icon="🚘", layout="wide")
st.markdown(
    """
    <style>
      .app-container {max-width: 900px; margin:auto; padding:1rem;}
      .title {font-size:3rem; font-weight:700; color:#1f4e79; text-align:center; margin-bottom:0.5rem;}
      .subtitle {font-size:1.2rem; text-align:center; margin-bottom:2rem; color:#33475b;}
      .question-btn button {
        width:100%; text-align:left; padding:0.75rem 1rem;
        border-radius:0.5rem; background-color:#2563eb; color:#ffffff;
        box-shadow:0 2px 6px rgba(0,0,0,0.15); margin-bottom:0.5rem;
      }
      .stChatMessage.user > div {background-color:#f0f9ff; border-radius:1rem 1rem 1rem 0.25rem; padding:0.75rem;}
      .stChatMessage.assistant > div {background-color:#eef2ff; border-radius:1rem 1rem 0.25rem 1rem; padding:0.75rem;}
      .stChatMessage {margin-bottom:0.5rem;}
      .footer {font-size:0.8rem; text-align:center; color:#888888; margin-top:2rem;}
    </style>
    <div class="app-container">
    """,
    unsafe_allow_html=True,
)

# Initialize chain 
@st.cache_resource(show_spinner="🔄 Initializing bot...")
def init_chain() -> Optional[RetrievalQA]:
    load_dotenv()
    settings = Settings()
    try:
        store = load_vectorstore(settings.chroma_dir)
    except FileNotFoundError:
        return None
    llm = ChatOpenAI(model=settings.llm_model, temperature=0.1, max_tokens=1000)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": settings.retrieval_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": AUTOMOTIVE_PROMPT},
    )

qa_chain = init_chain()

# Header & Samples
st.markdown('<div class="title">🚘 Car Governance Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask real-world automotive governance and safety questions</div>', unsafe_allow_html=True)

st.markdown('**Sample Engineering Questions**')
sample_questions = [
    "Which ASPICE process area covers system requirements engineering?",
    "How is ISO 26262 integrated into an agile development workflow?",
    "What are the key activities for ISO 26262 functional safety concept?",
    "How do ASIL classifications influence hardware safety design?",
    "Explain traceability requirements in ASPICE for suppliers.",
]
for q in sample_questions:
    if st.button(q, key=q, use_container_width=True):
        st.session_state['user_input'] = q

st.divider()

# Chat state 
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Render history
for msg in st.session_state['history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'], unsafe_allow_html=True)

# Input area
user_input = st.chat_input("Type your question…")
# Override if sample clicked
if not user_input and st.session_state.get('user_input'):
    user_input = st.session_state.pop('user_input')

if user_input:
    st.session_state['history'].append({'role': 'user', 'content': user_input})
    st.chat_message('user').markdown(user_input)

    if qa_chain is None:
        err = "Vector store missing. Run embed_store.py first."
        st.error(err)
        st.session_state['history'].append({'role':'assistant','content':err})
    else:
        with st.chat_message('assistant'):
            with st.spinner("Thinking…"):
                try:
                    res = qa_chain.invoke({'query': user_input})
                    ans = res['result']
                    srcs = res.get('source_documents', [])
                    st.markdown(ans)
                    with st.expander("📎 Sources"):
                        for d in srcs:
                            meta = d.metadata
                            st.markdown(f"- {meta.get('source','?')} • p.{meta.get('page','?')}")
                    st.session_state['history'].append({'role':'assistant','content':ans})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state['history'].append({'role':'assistant','content':f"Error: {e}"})

# Footer & close container
st.markdown(f'<div class="footer">Built {dt.datetime.now():%Y-%m-%d %H:%M:%S} – Car Governance RAG</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)