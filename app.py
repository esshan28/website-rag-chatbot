import os
os.environ["USER_AGENT"] = "website-rag-chatbot/1.0"

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.set_num_threads(os.cpu_count())

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Website RAG Chatbot", page_icon="🌐")
st.title("🌐 Website RAG Chatbot")
st.caption("Paste a website URL, then ask questions about its content.")

# ---------------------------
# LOAD MODELS (cached so they only load once)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return tokenizer, model

embeddings = load_embeddings()
tokenizer, model = load_llm()

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

                # Build prompt using chat template
                system_msg = (
                    "You are a helpful assistant that answers questions about a website "
                    "using only the provided context. If the answer is not in the context, "
                    "say you don't know."
                )
                user_msg = f"Context:\n{context}\n\nQuestion: {question}"

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                )
                input_ids = inputs["input_ids"]

                # Generate answer
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Only decode the newly generated tokens
                new_tokens = output_ids[0][input_ids.shape[-1]:]
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

                st.write(answer)

                with st.expander("🔍 Retrieved context (debug)"):
                    st.write(context)

        st.session_state.messages.append({"role": "assistant", "content": answer})