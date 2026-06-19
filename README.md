# 📄 Research Paper Q&A — Corrective RAG Agent

A production-ready AI agent that lets you upload any research paper (PDF) and ask questions about it in natural language. Built with **Corrective RAG (CRAG)**, **LangGraph**, **Gemini embeddings**, and **Groq LLaMA** — with real-time answer quality scoring via **LangSmith**.
  
📁 **GitHub:** [github.com/himanshi20004/research-paper-rag](https://github.com/himanshi20004/research-paper-rag)

---

## 🧠 What makes this different from basic RAG?

Most RAG systems blindly use whatever chunks they retrieve — even if those chunks are irrelevant to the question. This project implements **Corrective RAG (CRAG)**, which adds an intelligent self-correction step:

| Step | Basic RAG | Corrective RAG (this project) |
|------|-----------|-------------------------------|
| Retrieve chunks | ✅ | ✅ |
| Grade each chunk for relevance | ❌ | ✅ |
| Fall back to web search if chunks are bad | ❌ | ✅ |
| Score answer quality after generation | ❌ | ✅ |
| Log traces to LangSmith | ❌ | ✅ |

---

## ⚙️ How it works

```
User Question
      ↓
ChromaDB / FAISS — retrieve top-4 similar chunks
      ↓
LLM grades each chunk → relevant or irrelevant?
      ↓
   [If irrelevant]          [If relevant]
   Tavily Web Search    →   Skip web search
           ↓                      ↓
      Combine context  ←──────────┘
           ↓
   Generate answer (Groq LLaMA)
           ↓
   Evaluate: Faithfulness + Relevance scores
           ↓
   Log full trace to LangSmith
           ↓
   Show answer + confidence score to user
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Orchestration | LangGraph | Stateful agent graph with conditional routing |
| LLM | Groq (LLaMA 3.1 8B) | Free tier, extremely fast (~0.5s responses) |
| Embeddings | Gemini Embedding-001 | High quality, free tier |
| Vector Store | FAISS | Lightweight, no server needed, works on Windows |
| Web Search | Tavily API | Clean search results, 1000 free calls/month |
| Observability | LangSmith | Traces every step, logs eval scores |
| Frontend | Streamlit | Clean UI with confidence score cards |
| PDF Parsing | PyPDF | Reliable page-by-page extraction |

---

## ✨ Features

- 📤 **Upload any PDF** research paper and start asking questions instantly
- 🔍 **Corrective RAG** — grades retrieved chunks before generating answers
- 🌐 **Automatic web search fallback** via Tavily when PDF chunks are insufficient
- 📊 **Real-time scoring** — Faithfulness, Relevance, and Confidence % shown after every answer
- 🟢 **Source badge** — tells you whether the answer came from PDF or web search
- 🔬 **LangSmith integration** — full observability: traces, token counts, latency per step
- 💬 **Multi-turn chat** — remembers previous questions in the session
- 🚀 **Deployed on Streamlit Cloud** — publicly accessible, no setup needed

---

## 🚀 Run locally

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/himanshi20004/research-paper-rag.git
cd research-paper-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research-paper-rag
```

Get your free keys here:
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)
- **Groq** → [console.groq.com](https://console.groq.com)
- **Tavily** → [tavily.com](https://tavily.com)
- **LangSmith** → [smith.langchain.com](https://smith.langchain.com)

### 5. Run the app
```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project structure

```
research-paper-rag/
├── app/
│   ├── __init__.py
│   ├── ingestor.py       # PDF loading, chunking, FAISS embedding
│   ├── retriever.py      # Similarity search from FAISS
│   ├── grader.py         # LLM grades each chunk for relevance
│   ├── web_search.py     # Tavily fallback search
│   ├── generator.py      # Final answer generation with Groq
│   ├── evaluator.py      # Faithfulness + relevance scoring
│   └── graph.py          # LangGraph CRAG state machine
├── ui/
│   └── streamlit_app.py  # Streamlit frontend
├── .env                  # API keys (never commit this)
├── .gitignore
├── packages.txt          # System dependencies for Streamlit Cloud
├── requirements.txt
└── README.md
```

---

## 📊 LangSmith Observability

Every query is fully traced in LangSmith — you can see:
- Which chunks were retrieved and their relevance grades
- Whether web search was triggered
- Token counts and latency for each step
- Faithfulness and relevance scores logged as evaluation metrics

View your traces at [smith.langchain.com](https://smith.langchain.com) under the project `research-paper-rag`.

---

## 🔑 API Rate Limits (Free Tier)

| Service | Free Limit |
|---------|-----------|
| Gemini Embedding | 1,500 requests/day |
| Groq LLaMA 3.1 8B | 14,400 requests/day, 30 req/min |
| Tavily Search | 1,000 searches/month |
| LangSmith | Unlimited traces (free tier) |

---


