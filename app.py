import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

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
api_key = os.getenv("GEMINI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="DocuMind AI — Smart Document Q&A",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREMIUM UI — CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-base:       #0a0a1a;
        --bg-surface:    #0f0f24;
        --bg-elevated:   #161633;
        --bg-glass:      rgba(22, 22, 51, 0.65);
        --bg-glass-2:    rgba(30, 30, 65, 0.5);
        --border:        rgba(255, 255, 255, 0.06);
        --border-hover:  rgba(255, 255, 255, 0.12);
        --accent:        #6C63FF;
        --accent-2:      #00D4AA;
        --accent-glow:   rgba(108, 99, 255, 0.15);
        --accent-glow-2: rgba(0, 212, 170, 0.12);
        --danger:        #FF6B6B;
        --text:          #E8E8F0;
        --text-2:        #A0A0BE;
        --text-3:        #5C5C7A;
        --radius:        14px;
        --radius-sm:     10px;
        --shadow:        0 4px 24px rgba(0,0,0,0.4);
        --shadow-lg:     0 8px 40px rgba(0,0,0,0.5);
    }

    * { font-family: 'Inter', -apple-system, sans-serif !important; }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* ── App background ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-base) !important;
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(108, 99, 255, 0.08), transparent),
            radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0, 212, 170, 0.05), transparent) !important;
    }
    .main .block-container {
        padding: 2rem 1.5rem !important;
        max-width: 1100px !important;
    }

    /* ── Hide chrome ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* ── Sidebar toggle ── */
    [data-testid="collapsedControl"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
        color: var(--text) !important;
    }

    /* ━━━━ SIDEBAR ━━━━ */
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 1.1rem 0 !important;
    }

    /* Sidebar brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.2rem 0 1rem 0;
    }
    .sidebar-brand-icon {
        width: 36px; height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .sidebar-brand-text {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar section label */
    .section-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: var(--text-3) !important;
        margin-bottom: 0.7rem;
        padding-left: 1px;
    }

    /* Status chips */
    .status-chip {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .status-online {
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.18);
        color: var(--accent-2) !important;
    }
    .status-offline {
        background: rgba(255, 107, 107, 0.08);
        border: 1px solid rgba(255, 107, 107, 0.18);
        color: var(--danger) !important;
    }
    .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-on {
        background: var(--accent-2);
        box-shadow: 0 0 6px var(--accent-2);
        animation: blink 2.5s ease-in-out infinite;
    }
    .dot-off {
        background: var(--danger);
        box-shadow: 0 0 6px var(--danger);
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    /* Model badge */
    .model-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        background: var(--accent-glow);
        border: 1px solid rgba(108, 99, 255, 0.15);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent) !important;
    }

    /* Sidebar inputs */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: var(--bg-glass-2) !important;
        border: 1px dashed rgba(108, 99, 255, 0.2) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.8rem !important;
        transition: border-color 0.3s !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider p {
        font-size: 0.78rem !important;
        color: var(--text-2) !important;
    }

    /* ━━━━ HERO ━━━━ */
    .hero {
        background: var(--bg-glass);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.8rem 2rem 2.2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent-2), transparent);
        background-size: 200% 100%;
        animation: sweep 4s linear infinite;
    }
    @keyframes sweep {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .hero-badge {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: var(--accent);
        background: var(--accent-glow);
        padding: 5px 14px;
        border-radius: 20px;
        border: 1px solid rgba(108, 99, 255, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--text);
        margin: 0 0 0.5rem 0;
    }
    .hero h1 .gradient {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 0.95rem;
        color: var(--text-2);
        margin: 0;
        line-height: 1.5;
    }

    /* ━━━━ METRIC CARDS ━━━━ */
    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.1rem;
        margin-bottom: 0.7rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }
    .metric-val.accent { color: var(--accent); }
    .metric-val.green  { color: var(--accent-2); }
    .metric-lbl {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-3);
        margin-top: 0.15rem;
    }

    /* ━━━━ EMPTY STATE ━━━━ */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        background: var(--bg-glass);
        border: 1px dashed rgba(108, 99, 255, 0.15);
        border-radius: 20px;
        margin: 1rem 0;
    }
    .empty-state .icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.7;
    }
    .empty-state h3 {
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }
    .empty-state p {
        color: var(--text-3);
        font-size: 0.85rem;
        margin: 0;
    }

    /* ━━━━ SUCCESS BANNER ━━━━ */
    .banner-success {
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--accent-glow-2);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: var(--radius-sm);
        padding: 0.9rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--accent-2);
    }

    /* ━━━━ CHAT ━━━━ */
    [data-testid="stChatMessage"] {
        background: var(--bg-glass) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.3rem !important;
        margin-bottom: 0.5rem !important;
        animation: msgIn 0.3s ease-out;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] strong {
        color: var(--text) !important;
    }
    [data-testid="stChatMessage"] code {
        background: var(--bg-elevated) !important;
        color: var(--accent-2) !important;
        padding: 2px 6px !important;
        border-radius: 5px !important;
    }
    @keyframes msgIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-surface) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        caret-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-3) !important;
    }

    /* ━━━━ BUTTONS ━━━━ */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), #8B7CFF) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(108, 99, 255, 0.35) !important;
    }

    /* ━━━━ SLIDER ━━━━ */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* ━━━━ ALERTS ━━━━ */
    .stAlert { border-radius: var(--radius-sm) !important; }

    /* ━━━━ SPINNER ━━━━ */
    .stSpinner > div { color: var(--accent) !important; }

    /* ━━━━ FOOTER ━━━━ */
    .app-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.2rem 0;
        border-top: 1px solid var(--border);
    }
    .app-footer p {
        color: var(--text-3);
        font-size: 0.72rem;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .app-footer span.sep {
        display: inline-block;
        margin: 0 8px;
        color: var(--border-hover);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    # Brand
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">✦</div>
            <span class="sidebar-brand-text">DocuMind AI</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API status
    st.markdown('<p class="section-label">Connection</p>', unsafe_allow_html=True)
    if api_key:
        st.markdown(
            '<div class="status-chip status-online">'
            '<span class="dot dot-on"></span> Gemini API Connected'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-chip status-offline">'
            '<span class="dot dot-off"></span> API Key Missing'
            '</div>',
            unsafe_allow_html=True
        )
        st.warning("Add GEMINI_API_KEY to .env file")

    st.markdown("---")

    # Upload
    st.markdown('<p class="section-label">Document</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], label_visibility="collapsed")

    st.markdown("---")

    # Model + Settings
    st.markdown('<p class="section-label">Engine</p>', unsafe_allow_html=True)
    model_name = "gemini-2.5-flash"
    st.markdown(
        '<div class="model-badge">⚡ Gemini 2.5 Flash</div>',
        unsafe_allow_html=True
    )
    temperature = st.slider("Creativity", 0.0, 1.0, 0.0, 0.1,
                            help="Higher = more creative, lower = more factual")
    max_results = st.slider("Context depth", 1, 20, 10,
                            help="Number of document chunks to use for context")

    st.markdown("---")

    # Clear
    if st.session_state.chat_history:
        if st.button("✕  Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        '<p style="font-size:0.7rem; color:var(--text-3) !important; text-align:center; '
        'margin-top:1.5rem;">v1.0 · Built with LangChain</p>',
        unsafe_allow_html=True
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE LOGIC (unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner=False)
def load_and_process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, len(docs), None
    except Exception as e:
        return None, 0, str(e)

def ask_question(question, vectorstore, temperature, max_results, model_name):
    if not api_key:
        return "⚠️ API key not configured. Add GEMINI_API_KEY to your .env file."
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": max_results}
        )
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are a helpful assistant. Answer the question based on the context below.
If you don't know, say so clearly.

Context: {context}

Question: {question}

Answer:"""
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Hero
st.markdown("""
    <div class="hero">
        <div class="hero-badge">Retrieval-Augmented Generation</div>
        <h1>Ask your <span class="gradient">documents</span> anything</h1>
        <p>Upload a PDF, and get precise, AI-powered answers grounded in your content.</p>
    </div>
""", unsafe_allow_html=True)

# Layout
col_main, col_side = st.columns([3, 1])

with col_side:
    if st.session_state.vectorstore:
        st.markdown(
            '<div class="metric-card">'
            '<p class="metric-val green">●  Ready</p>'
            '<p class="metric-lbl">Document Status</p>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-val accent">{st.session_state.get("num_chunks", 0)}</p>'
            f'<p class="metric-lbl">Indexed Chunks</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        q_count = len(st.session_state.chat_history) // 2
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-val">{q_count}</p>'
            f'<p class="metric-lbl">Questions Asked</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
            <div class="empty-state">
                <div class="icon">📄</div>
                <h3>No document loaded</h3>
                <p>Upload a PDF from the sidebar to start asking questions.</p>
            </div>
        """, unsafe_allow_html=True)

with col_main:
    # Process uploaded file
    if uploaded_file:
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Processing document..."):
            vectorstore, num_chunks, error = load_and_process_pdf("temp_upload.pdf")
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.markdown(
                    f'<div class="banner-success">'
                    f'✓ Document indexed — <strong>{num_chunks} chunks</strong> ready for Q&A'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error(f"Failed to process PDF: {error}")

    elif os.path.exists("my_paper.pdf") and not st.session_state.vectorstore:
        with st.spinner("Loading default document..."):
            vectorstore, num_chunks, error = load_and_process_pdf("my_paper.pdf")
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.num_chunks = num_chunks
                st.markdown(
                    f'<div class="banner-success">'
                    f'✓ Default document loaded — <strong>{num_chunks} chunks</strong> ready'
                    f'</div>',
                    unsafe_allow_html=True
                )

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your document..."):
    if not st.session_state.vectorstore:
        st.error("Upload a PDF first to start asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(
                    prompt,
                    st.session_state.vectorstore,
                    temperature,
                    max_results,
                    model_name
                )
                st.write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
    <div class="app-footer">
        <p>
            DocuMind AI
            <span class="sep">·</span> Streamlit
            <span class="sep">·</span> LangChain
            <span class="sep">·</span> Google Gemini
        </p>
    </div>
""", unsafe_allow_html=True)