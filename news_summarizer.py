import streamlit as st
import os
import time
import tempfile
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from newsapi import NewsApiClient
class ManualDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
# -------------------- Load API Keys --------------------
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
news_api_key = os.getenv('NEWSAPI_KEY')
news_client = None
if news_api_key:
    news_client = NewsApiClient(api_key=news_api_key)

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("NEWS & Document Summarizer Chatbot")

# -------------------- Session State --------------------
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# -------------------- Embeddings --------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# -------------------- Document Processing --------------------
def process_documents(uploaded_files, manual_text=None):
    with st.spinner("Processing documents..."):
        try:
            embeddings = get_embeddings()
            all_docs = []

            # Handle uploaded files
            for file in uploaded_files:
                suffix = os.path.splitext(file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(file.read())
                    temp_path = temp_file.name

                if suffix == ".pdf":
                    loader = PyPDFLoader(temp_path)
                elif suffix == ".txt":
                    loader = TextLoader(temp_path)
                elif suffix == ".csv":
                    loader = CSVLoader(temp_path)
                elif suffix in [".docx", ".doc"]:
                    loader = UnstructuredWordDocumentLoader(temp_path)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = file.name
                all_docs.extend(docs)
                os.unlink(temp_path)

            # Handle manual input
            if manual_text:
                all_docs.append(ManualDocument(page_content=manual_text, metadata={"source": "manual_input"}))

            if not all_docs:
                st.error("No valid documents to process.")
                return None, None

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)

            # Create FAISS vector store
            vector_store = FAISS.from_documents(splits, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            return vector_store, retriever

        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return None, None

# -------------------- LLM --------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

# -------------------- RAG Chain --------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided documents.

    Context: {context}

    Question: {question}

    Answer the question based only on the provided context. If the information is not in the 
    context, say "I don't have enough information to answer this question based on the provided documents."

    Provide a clear, concise, and informative answer.
    """)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Word, TXT, or CSV files",
        accept_multiple_files=True,
        type=['pdf','txt','csv','docx','doc']
    )
    manual_text = st.text_area("Or enter text manually")
    news_topic = st.text_input("Enter topic to fetch news")
    max_articles = st.number_input("Max articles", min_value=1, max_value=20, value=5, step=1)

    if st.button("Fetch News"):
        if not news_client:
            st.error("No NEWS_API_KEY found. Please add it to your .env file.")
        elif not news_topic.strip():
            st.warning("Please enter a topic to fetch news.")
        else:
            with st.spinner(f"Fetching news about '{news_topic}'..."):
                try:
                    articles = news_client.get_everything(
                        q=news_topic,
                        language='en',
                        page_size=max_articles,
                        sort_by='relevancy'
                    )['articles']

                    if not articles:
                        st.warning("No news articles found for this topic.")
                    else:
                        news_docs = []
                        for article in articles:
                            content = article.get('content') or article.get('description') or ''
                            if content:
                                news_docs.append(ManualDocument(
                                    page_content=content,
                                    metadata={'source': article.get('url', 'Unknown')}
                                ))

                        if news_docs:
                                embeddings = get_embeddings()
                                # Split news articles into chunks
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                                splits = text_splitter.split_documents(news_docs)

                                # Create FAISS vector store
                                vector_store = FAISS.from_documents(splits, embeddings)
                                retriever = vector_store.as_retriever(search_kwargs={"k": 5})

                                st.session_state.vector_store = vector_store
                                st.session_state.retriever = retriever
                                st.session_state.processed_docs = True
                                st.success(f"✅ Fetched and processed {len(news_docs)} news articles")
                except Exception as e:
                    st.error(f"Error fetching news: {e}")

    if st.button("Process Documents"):
        vector_store, retriever = process_documents(uploaded_files, manual_text)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.processed_docs = True
            st.success(f"✅ Documents processed successfully")
        else:
            st.error("Failed to process documents")

    if st.session_state.processed_docs:
        st.sidebar.success("Documents ready for questions")
    else:
        st.sidebar.info("Please upload/process documents first")

    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
# -------------------- Main Chat --------------------
st.markdown("### Ask questions about your documents")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not st.session_state.processed_docs:
        response = "Please upload and process documents first."
        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                rag_chain, retriever = create_rag_chain(st.session_state.retriever)
                start_time = time.time()
                response_text = rag_chain.invoke(prompt)
                elapsed = time.time() - start_time
                placeholder.write(response_text)
                st.caption(f"Response time: {elapsed:.2f}s")
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                # Show sources
                relevant_docs = retriever.invoke(prompt)
                with st.expander("Document Sources"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Source {i+1}**: {doc.metadata.get('source','Unknown')}")
                        st.markdown(f"{doc.page_content[:300]}...")
                        st.markdown("---")
            except Exception as e:
                placeholder.error(f"Error: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": str(e)})

if not st.session_state.processed_docs and not st.session_state.chat_history:
    st.info("👈 Upload documents or enter text in the sidebar to get started")