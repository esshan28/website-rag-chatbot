import os
os.environ["USER_AGENT"] = "website-rag-chatbot/1.0"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from huggingface_hub import InferenceClient

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Website RAG Chatbot", page_icon="🌐")
st.title("🌐 Website RAG Chatbot")
st.caption("Paste a website URL, then ask questions about its content.")

# ---------------------------
# LOAD MODELS (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_hf_client():
    hf_token = os.environ.get("HF_TOKEN")
    return InferenceClient(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        token=hf_token
    )

embeddings = load_embeddings()
hf_client = load_hf_client()

# ---------------------------
# SESSION STATE
# ---------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

# ---------------------------
# SIDEBAR: URL INPUT
# ---------------------------
with st.sidebar:
    st.header("Load a website")
    url = st.text_input("Website URL", placeholder="https://example.com")

    if st.button("Load Website", type="primary"):
        if not url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Loading and processing website..."):
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    docs = splitter.split_documents(docs)

                    st.session_state.vector_db = InMemoryVectorStore.from_documents(
                        documents=docs,
                        embedding=embeddings
                    )
                    st.session_state.loaded_url = url
                    st.session_state.messages = []
                    st.success(f"Loaded {len(docs)} chunks from the website!")
                except Exception as e:
                    st.error(f"Failed to load website: {e}")

    if st.session_state.loaded_url:
        st.info(f"Currently loaded:\n{st.session_state.loaded_url}")

# ---------------------------
# CHAT DISPLAY
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# CHAT INPUT
# ---------------------------
if st.session_state.vector_db is None:
    st.info("👈 Load a website from the sidebar to start chatting.")
else:
    question = st.chat_input("Ask something about the website...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                records = st.session_state.vector_db.similarity_search(question, k=3)
                context = "\n\n".join(doc.page_content for doc in records)

                # HF Inference API call
                response = hf_client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that answers questions "
                                "about a website using only the provided context. "
                                "If the answer is not in the context, say you don't know."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}"
                        }
                    ],
                    max_tokens=256,
                )

                answer = response.choices[0].message.content

                st.write(answer)

                with st.expander("🔍 Retrieved context (debug)"):
                    st.write(context)

        st.session_state.messages.append({"role": "assistant", "content": answer})
