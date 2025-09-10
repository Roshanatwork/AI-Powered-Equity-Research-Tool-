import os
import streamlit as st
import pickle
import time
import requests
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings

# FIX: Add the required asyncio fix for Streamlit
asyncio.set_event_loop(asyncio.new_event_loop())

# --- Session State Initialization ---
st.session_state.setdefault('urls', ["", "", ""])
st.session_state.setdefault('vectorstore', None)
st.session_state.setdefault('llm', None)
# Initialize embeddings once and store in session state
if 'embeddings' not in st.session_state or st.session_state.embeddings is None:
    # This will download the model from Hugging Face the first time it's run
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


st.title("AI-Powered Equity Research Tool 📈")
st.sidebar.title("Configuration")

# --- API Key Input (Only for Gemini Q&A) ---
api_key = st.sidebar.text_input("Google Gemini API Key (for Q&A only)", type="password")
if api_key and st.session_state.llm is None:
    try:
        os.environ['GOOGLE_API_KEY'] = api_key
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        st.sidebar.success("Gemini API Key accepted for Q&A.")
    except Exception as e:
        st.sidebar.error(f"API Key Error: {e}")
        st.session_state.llm = None
elif not api_key:
    st.sidebar.info("Enter a Gemini API key to enable AI-powered answers.")
    st.session_state.llm = None

# --- URL Inputs & Buttons ---
st.sidebar.title("News Article URLs")
for i in range(3):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i])

process_url_clicked = st.sidebar.button("Process URLs")
clear_cache_clicked = st.sidebar.button("Clear Cache")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

def validate_urls(urls):
    valid_urls = []
    for url in urls:
        if url.strip():
            try:
                response = requests.head(url, timeout=5, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    valid_urls.append(url)
                else:
                    st.sidebar.warning(f"Skipping {url} (Status: {response.status_code})")
            except requests.RequestException:
                st.sidebar.warning(f"Skipping {url} (Unreachable)")
    return valid_urls

if clear_cache_clicked and os.path.exists(file_path):
    os.remove(file_path)
    st.session_state.vectorstore = None
    st.sidebar.success("Cache cleared!")

if process_url_clicked:
    urls_to_process = [url for url in st.session_state.urls if url.strip()]
    if not urls_to_process:
        st.sidebar.error("Please provide at least one URL.")
    else:
        try:
            main_placeholder.text("Validating URLs...")
            valid_urls = validate_urls(urls_to_process)

            if not valid_urls:
                st.sidebar.error("No valid or accessible URLs to process.")
            else:
                main_placeholder.text("Loading documents...✅")
                loader = WebBaseLoader(valid_urls)
                data = loader.load()

                main_placeholder.text("Splitting text...✅")
                # FIX: Use sensible separators for the text splitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(data)

                main_placeholder.text("Creating embeddings (using FREE local model)...")
                # FIX: Removed the flawed and unnecessary batching logic
                vectorstore = FAISS.from_documents(docs, st.session_state.embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                st.session_state.vectorstore = vectorstore
                main_placeholder.text("Processing complete! ✅")
                st.sidebar.success(f"Processed {len(valid_urls)} URL(s).")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Load vector store from file if it exists
if os.path.exists(file_path) and st.session_state.vectorstore is None:
    with open(file_path, "rb") as f:
        st.session_state.vectorstore = pickle.load(f)

# --- Query Section ---
query = main_placeholder.text_input("Ask a question about the news articles:")
if query:
    if st.session_state.vectorstore is None:
        st.error("Please process URLs to build the knowledge base first.")
    else:
        try:
            if st.session_state.llm:
                # Use Gemini for Q&A if API key is provided
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=st.session_state.llm,
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)
            else:
                # Fallback: just show the retrieved documents
                docs = st.session_state.vectorstore.similarity_search(query, k=3)
                result = {
                    "answer": "Gemini API key not provided. Here are the most relevant document excerpts:",
                    "sources": "\n\n".join([f"**Source:** {doc.metadata.get('source', 'N/A')}\n\n"
                                          f"{doc.page_content[:300]}..." for doc in docs])
                }
            
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                st.write(sources)
        except Exception as e:
            st.error(f"An error occurred during Q&A: {e}")
