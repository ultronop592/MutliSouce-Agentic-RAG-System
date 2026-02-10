import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Changed to GEMINI_API_KEY

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed CSS - TEXT NOW VISIBLE
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title container */
    .title-box {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    /* Stats box */
    .stats-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #2d3748;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Success message */
    .success-msg {
        background: #48bb78;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key status (hidden value)
    if api_key:
        st.markdown("🟢 **API Connected**")
    else:
        st.markdown("🔴 **API Not Found**")
        st.warning("Add GEMINI_API_KEY to .env file")
    
    st.markdown("---")
    
    # File uploader
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose PDF", type=['pdf'], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Model settings
    st.markdown("### 🎛️ Model Settings")
    model_name = st.selectbox(
        "Select Model",
        ["models/gemini-1.5-flash-latest", "models/gemini-1.5-pro-latest", "models/gemini-2.0-flash"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_results = st.slider("Retrieved Chunks", 1, 20, 10)
    
    st.markdown("---")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("💡 *Upload a PDF to get started!*")

# Function to load and process PDF
@st.cache_resource(show_spinner=False)
def load_and_process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, len(docs), None
    except Exception as e:
        return None, 0, str(e)

# Function to get answer
def ask_question(question, vectorstore, temperature, max_results, model_name):
    if not api_key:
        return "⚠️ API key not configured. Please add GEMINI_API_KEY to your .env file."
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": max_results}
        )
        
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the context below. If you don't know, say so clearly.

Context: {context}

Question: {question}

Answer:"""
        
        # Using your specified model
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Main content
st.markdown("""
    <div class="title-box">
        <h1 style="margin:0; font-size:3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🤖 RAG Q&A Assistant
        </h1>
        <p style="margin:0.5rem 0 0 0; color:#666; font-size:1.2rem;">
            Ask intelligent questions about your documents
        </p>
    </div>
""", unsafe_allow_html=True)

# Two column layout
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
    if st.session_state.vectorstore:
        st.markdown(f'<p class="stat-value">✅</p><p class="stat-label">Ready</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="stat-value">{st.session_state.get("num_chunks", 0)}</p><p class="stat-label">Chunks Loaded</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="stat-value">⏳</p><p class="stat-label">Waiting for PDF</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    # Process uploaded file
    if uploaded_file:
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("🔄 Processing your document..."):
            vectorstore, num_chunks, error = load_and_process_pdf("temp_upload.pdf")
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.markdown(f'<div class="success-msg">✅ Document processed! {num_chunks} chunks created.</div>', unsafe_allow_html=True)
            else:
                st.error(f"Failed to process PDF: {error}")
    
    # Load default PDF
    elif os.path.exists("my_paper.pdf") and not st.session_state.vectorstore:
        with st.spinner("🔄 Loading default document..."):
            vectorstore, num_chunks, error = load_and_process_pdf("my_paper.pdf")
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.markdown(f'<div class="success-msg">✅ Default document loaded! {num_chunks} chunks created.</div>', unsafe_allow_html=True)

# Display chat history with proper styling
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])  # Using st.write for better text rendering

# Chat input
if prompt := st.chat_input("💬 Ask me anything about the document..."):
    if not st.session_state.vectorstore:
        st.error("⚠️ Please upload a PDF document first!")
    else:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = ask_question(
                    prompt,
                    st.session_state.vectorstore,
                    temperature,
                    max_results,
                    model_name
                )
                st.write(response)  # Using st.write instead of st.markdown
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
    <div style="text-align:center; padding:2rem; color:white;">
        <p style="margin:0;">Built with ❤️ using Streamlit, LangChain & Google Gemini</p>
    </div>
""", unsafe_allow_html=True)