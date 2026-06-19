import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestor import ingest_pdf
from app.graph import run_query

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS for a clean look ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }
    
    .title-area {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .title-area h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    .title-area p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .score-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .score-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .score-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .score-green  { color: #4ade80; }
    .score-yellow { color: #facc15; }
    .score-red    { color: #f87171; }

    .answer-box {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-web  { background: #1e3a5f; color: #60a5fa; }
    .badge-pdf  { background: #1a3a2a; color: #4ade80; }

    .stButton>button {
        background: #7c3aed;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { background: #6d28d9; }

    .chunk-card {
        background: #151922;
        border-left: 3px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-area">
    <h1>📄 Research Paper Q&A</h1>
    <p>Upload a PDF research paper and ask questions — powered by Corrective RAG + Gemini</p>
</div>
""", unsafe_allow_html=True)

# ── Layout: sidebar + main ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload Paper")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        help="Upload any research paper in PDF format"
    )

    if uploaded_file and not st.session_state.pdf_ingested:
        with st.spinner("Processing PDF..."):
            # Save to temp file (Streamlit uploads are in-memory)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            num_chunks = ingest_pdf(tmp_path)
            os.unlink(tmp_path)  # clean up temp file
            
            st.session_state.pdf_ingested = True
            st.session_state.pdf_name = uploaded_file.name

        st.success(f"✅ Ingested {num_chunks} chunks")

    if st.session_state.pdf_ingested:
        st.markdown(f"**Active paper:**  \n`{st.session_state.get('pdf_name', 'unknown')}`")
        if st.button("Upload new paper"):
            st.session_state.pdf_ingested = False
            st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. Your PDF is chunked + embedded with Gemini
2. Your question retrieves top-4 chunks
3. Each chunk is **graded** for relevance
4. Irrelevant chunks → Tavily web search
5. Answer is generated + **scored** for accuracy
6. All runs logged to **LangSmith**
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.pdf_ingested:
    st.info("👈 Upload a research paper PDF from the sidebar to get started.")
else:
    # Show chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
            
            # Scores row
            scores = item.get("scores", {})
            if scores:
                c1, c2, c3 = st.columns(3)
                
                def color_class(val):
                    if val >= 8: return "score-green"
                    if val >= 5: return "score-yellow"
                    return "score-red"

                with c1:
                    v = scores.get("faithfulness", 0)
                    st.markdown(f'<div class="score-card"><div class="label">Faithfulness</div><div class="value {color_class(v)}">{v}/10</div></div>', unsafe_allow_html=True)
                with c2:
                    v = scores.get("relevance", 0)
                    st.markdown(f'<div class="score-card"><div class="label">Relevance</div><div class="value {color_class(v)}">{v}/10</div></div>', unsafe_allow_html=True)
                with c3:
                    v = scores.get("confidence", 0)
                    cls = "score-green" if v >= 75 else ("score-yellow" if v >= 50 else "score-red")
                    st.markdown(f'<div class="score-card"><div class="label">Confidence</div><div class="value {cls}">{v}%</div></div>', unsafe_allow_html=True)

            # Source badge
            badge = '<span class="badge badge-web">🌐 Used web search</span>' if item.get("used_web") else '<span class="badge badge-pdf">📄 PDF only</span>'
            st.markdown(badge, unsafe_allow_html=True)

    # Question input
    question = st.chat_input("Ask anything about your paper...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_query(question)
            
            answer = result.get("answer", "Sorry, I couldn't generate an answer.")
            scores = result.get("scores", {})
            used_web = result.get("used_web_search", False)

            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # Score cards
            if scores:
                c1, c2, c3 = st.columns(3)
                def color_class(val):
                    if val >= 8: return "score-green"
                    if val >= 5: return "score-yellow"
                    return "score-red"
                
                with c1:
                    v = scores.get("faithfulness", 0)
                    st.markdown(f'<div class="score-card"><div class="label">Faithfulness</div><div class="value {color_class(v)}">{v}/10</div></div>', unsafe_allow_html=True)
                with c2:
                    v = scores.get("relevance", 0)
                    st.markdown(f'<div class="score-card"><div class="label">Relevance</div><div class="value {color_class(v)}">{v}/10</div></div>', unsafe_allow_html=True)
                with c3:
                    v = scores.get("confidence", 0)
                    cls = "score-green" if v >= 75 else ("score-yellow" if v >= 50 else "score-red")
                    st.markdown(f'<div class="score-card"><div class="label">Confidence</div><div class="value {cls}">{v}%</div></div>', unsafe_allow_html=True)

            badge = '<span class="badge badge-web">🌐 Used web search</span>' if used_web else '<span class="badge badge-pdf">📄 PDF only</span>'
            st.markdown(badge, unsafe_allow_html=True)

            # Save to history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "scores": scores,
                "used_web": used_web
            })