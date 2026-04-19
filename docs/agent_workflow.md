# 🤖 HeliosCast — LangGraph Grid Optimization Agent: Workflow Documentation

> **Version:** 1.0 | **Framework:** LangGraph + LangChain + Google Gemini 2.5 Flash  
> **Generated:** April 2026

---

## 1. Overview

The HeliosCast Grid Optimization Agent is a **5-node agentic pipeline** built with [LangGraph](https://langchain-ai.github.io/langgraph/). It ingests hourly solar generation forecasts (produced by a Linear Regression ML model), reasons over the data using an LLM, retrieves grounding context from a FAISS vector store (RAG), and produces a professional Markdown grid optimization report.

### Core Objectives
- **Forecast** solar generation summary for the day
- **Assess** variability risks and grid stress periods
- **Retrieve** grounded industry protocols via RAG
- **Plan** concrete grid balancing, storage, and demand response actions
- **Report** a structured optimization strategy with references

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  HeliosCast Agent Pipeline                   │
│                                                             │
│  [forecast_records]                                         │
│        │                                                    │
│        ▼                                                    │
│  ┌─────────────────────┐                                    │
│  │  Forecaster_Analyzer │  ← ML output interpreter         │
│  └──────────┬──────────┘                                    │
│             │  forecast_summary                             │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │    Risk_Assessor    │  ← Rule-based + LLM risk flags    │
│  └──────────┬──────────┘                                    │
│             │  risk_assessment                              │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │    RAG_Retriever    │  ← FAISS vector store query       │
│  └──────────┬──────────┘                                    │
│             │  retrieved_docs                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │  Strategy_Planner   │  ← Hour-by-hour action plan       │
│  └──────────┬──────────┘                                    │
│             │  final_recommendations                        │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │  Report_Formatter   │  ← Markdown report assembler      │
│  └──────────┬──────────┘                                    │
│             │  report (Markdown)                            │
│             ▼                                               │
│           [END]                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Shared State: `AgentState`

All nodes communicate via a single `TypedDict` state object. No node holds private state — everything is passed through the graph.

| Field | Type | Set By | Description |
|-------|------|--------|-------------|
| `forecast_data` | `List[Dict]` | Input | Hourly records: `{timestamp, irradiance, cloud_cover, temperature, predicted_generation}` |
| `forecast_summary` | `str` | Forecaster_Analyzer | 3–4 sentence narrative of the generation profile |
| `risk_assessment` | `str` | Risk_Assessor | Risk level, flagged periods, primary drivers |
| `retrieved_docs` | `str` | RAG_Retriever | Relevant protocol excerpts from FAISS |
| `final_recommendations` | `str` | Strategy_Planner | Hour-by-hour Markdown action table + summaries |
| `report` | `str` | Report_Formatter | Final assembled Markdown report |
| `errors` | `List[str]` | Any node | Non-fatal error accumulator |

---

## 4. Node-by-Node Breakdown

### 4.1 `Forecaster_Analyzer`

**Purpose:** Translate raw ML predictions into a human-readable generation narrative.

**Input:** `forecast_data`

**Logic:**
1. Builds a pipe-table from all hourly records
2. Computes: total kWh, peak record, trough record
3. Calls Gemini 2.5 Flash with a structured prompt asking for a 3–4 sentence analysis highlighting: morning ramp, midday plateau, evening ramp-down

**Output → `forecast_summary`**

**Responsible AI Notes:**
- LLM is instructed to be specific with numbers, reducing hallucination risk
- Stats are pre-computed deterministically and injected into the prompt as ground truth

---

### 4.2 `Risk_Assessor`

**Purpose:** Identify periods of high variability and grid stress.

**Input:** `forecast_data`, `forecast_summary`

**Logic (hybrid rule-based + LLM):**

| Rule | Threshold | Flag |
|------|-----------|------|
| Irradiance drop | > 25% hour-over-hour | ⚠️ HIGH VARIABILITY |
| Cloud cover | > 60% | 🌩️ SEVERE OBSTRUCTION |
| Cloud cover | 40–60% | ☁️ MODERATE RISK |
| Near-zero generation | < 10 W during 08:00–16:00 | 🔴 CRITICAL |

The LLM then synthesises these flags into:
- Overall risk level (LOW / MEDIUM / HIGH / CRITICAL)
- Top 3 risk periods with timestamps
- Primary risk drivers

**Output → `risk_assessment`**

**Responsible AI Notes:**
- Rule-based pre-filtering ensures LLM cannot miss obvious flags
- Risk levels are bounded to a fixed enumeration (no free-form severity invented)

---

### 4.3 `RAG_Retriever`

**Purpose:** Ground recommendations in established industry protocols using Retrieval-Augmented Generation.

**Input:** `forecast_summary`, `risk_assessment`

**Logic:**
1. Constructs a composite retrieval query from the first 300 chars of each upstream node output
2. Queries the local FAISS vector store (`agent/faiss_index/`) with `k=5` chunks
3. Falls back to a hardcoded spinning-reserve guideline if FAISS fails

**Vector Store:**
- Built with `sentence-transformers` embeddings
- Knowledge base: `agent/knowledge_base/` (grid management documents, IEEE standards, IEA protocol excerpts)
- Index persisted to disk; loaded once via `@st.cache_resource`

**Output → `retrieved_docs`**

**Responsible AI Notes:**
- All retrieved content is sourced from pre-curated, domain-expert documents
- RAG grounding significantly reduces LLM fabrication of specific protocol numbers

---

### 4.4 `Strategy_Planner`

**Purpose:** Synthesise all upstream context into a concrete, actionable energy management plan.

**Input:** `forecast_data`, `forecast_summary`, `risk_assessment`, `retrieved_docs`

**Logic:**
Calls Gemini 2.5 Flash with a structured prompt requesting:
1. **Hour-by-hour action table** (Markdown) with:
   - `Action`: `CHARGE_BATTERY | DISCHARGE_BATTERY | EXPORT_GRID | IMPORT_GRID | STANDBY`
   - `Rationale` (one line)
   - `Priority`: HIGH / MEDIUM / LOW
2. **Battery Management Summary**
3. **Grid Interaction Summary**
4. **Demand Response Triggers**
5. **Reserve Activation Needs**

LLM is instructed to directly reference retrieved protocol documents.

**Output → `final_recommendations`**

**Responsible AI Notes:**
- Constrained action vocabulary prevents hallucinated non-standard operations
- Explicit instruction to cite retrieved docs provides verifiable reasoning chain

---

### 4.5 `Report_Formatter`

**Purpose:** Assemble all node outputs into a publish-ready Markdown document.

**Input:** All state fields

**Logic:**
Deterministically assembles the following report structure:

```
# ☀️ HeliosCast — Grid Optimization Report
## 1. Executive Summary (metrics table)
## 2. Forecast Analysis
## 3. Risk Assessment
## 4. Retrieved Industry Protocols (collapsible)
## 5. Grid Optimization Strategy & Recommendations
## 6. Appendix — Raw Forecast Data (full table)
```

**Output → `report`** (Markdown string, downloadable as `.md`)

---

## 5. Retrieval Integration & State Management

### RAG Integration
- FAISS index is loaded **once** from disk at app startup via `@st.cache_resource`
- The `retrieve_docs(query, k=5)` function in `agent/rag_setup.py` handles embedding + similarity search
- Retrieval query is composited from forecast + risk context to maximise relevance

### Streamlit Session State
All agent outputs and user settings are persisted in `st.session_state`:

| Key | Purpose |
|-----|---------|
| `agent_report` | Last generated Markdown report (survives tab switches) |
| `agent_running` | Boolean flag preventing double-execution |
| `google_api_key` | Persisted API key (written to `os.environ`) |
| `chat_history` | Chat turns for the Report Chatbot tab |
| `chat_context_set` | Tracks whether report was injected as chatbot context |

Chat history is **automatically reset** whenever a new report is generated, ensuring the chatbot always answers relative to the latest run.

---

## 6. Responsible AI Practices

| Practice | Implementation |
|----------|---------------|
| **Grounded generation** | RAG retrieval provides factual anchoring for all protocol references |
| **Constrained outputs** | Action vocabulary (`CHARGE_BATTERY`, etc.) is fixed in the prompt |
| **Hybrid reasoning** | Rule-based risk flags + LLM synthesis — critical checks cannot be skipped by the LLM |
| **Transparent context** | Full retrieved docs are displayed in the UI (collapsible section) |
| **Fallback handling** | FAISS errors fall back to hardcoded spinning-reserve guideline |
| **Deterministic stats** | Numerical inputs (kWh, peak, trough) are pre-computed, not hallucinated |
| **User-visible errors** | All pipeline errors are surfaced in the Streamlit UI status panel |
| **Report attribution** | Every report is timestamped and includes model + horizon metadata |

---

## 7. Deployment Notes

- **Environment:** Python 3.10+, Streamlit ≥ 1.35
- **Key dependencies:** `langgraph`, `langchain-google-genai`, `faiss-cpu`, `sentence-transformers`
- **API key:** Google Gemini API key set via sidebar widget (persisted to `os.environ`)
- **FAISS index:** Must be pre-built via `agent/rag_setup.py` before first run
- **Hosting:** Compatible with Streamlit Community Cloud, Hugging Face Spaces (with Git LFS for binary assets)

---

*HeliosCast — LangGraph Grid Optimization Agent | End-Sem Milestone 2 Documentation*
