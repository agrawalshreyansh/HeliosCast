"""
grid_agent.py — LangGraph-based Grid Optimization Agent for HeliosCast
=====================================================================
Workflow:
  forecast_data → Forecaster_Analyzer → Risk_Assessor
                  → RAG_Retriever → Strategy_Planner → Report_Formatter

Each node receives the full TypedDict State and returns a partial update.
The graph is compiled with LangGraph's StateGraph.
"""

from __future__ import annotations

import os
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

# --- LangGraph ---
from langgraph.graph import StateGraph, END

# --- LangChain LLM (Google Gemini via langchain-google-genai) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- Local RAG helper ---
from agent.rag_setup import retrieve_docs

# =====================================================================
# 0. LLM INITIALISATION
# =====================================================================
def _get_llm():
    """Returns a Gemini Flash LLM instance. Falls back gracefully."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable not set. "
            "Export it before running the agent."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )

# =====================================================================
# 1. STATE DEFINITION
# =====================================================================
class AgentState(TypedDict):
    """Shared mutable state passed between every LangGraph node."""

    # ── Inputs ─────────────────────────────────────────────────────────
    forecast_data: List[Dict[str, Any]]
    """List of hourly records: {timestamp, irradiance, cloud_cover,
       temperature, predicted_generation}"""

    # ── Node outputs ────────────────────────────────────────────────────
    forecast_summary: str
    """Plain-text summary produced by Forecaster_Analyzer."""

    risk_assessment: str
    """Risk periods and severity levels from Risk_Assessor."""

    retrieved_docs: str
    """Relevant protocol snippets fetched from FAISS by RAG_Retriever."""

    final_recommendations: str
    """Action plan from Strategy_Planner."""

    report: str
    """Fully formatted Markdown report from Report_Formatter."""

    # ── Metadata ────────────────────────────────────────────────────────
    errors: List[str]
    """Non-fatal errors / warnings accumulated during execution."""


# =====================================================================
# 2. NODE HELPERS
# =====================================================================

def _llm_call(system: str, human: str) -> str:
    """Wrapper: calls the LLM and returns the text response."""
    llm = _get_llm()
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    response = llm.invoke(messages)
    return response.content.strip()


def _format_forecast_table(records: List[Dict[str, Any]]) -> str:
    """Converts forecast list into a readable pipe-table string."""
    header = "| Hour | Irradiance (W/m²) | Cloud (%) | Temp (°C) | Generation (W) |"
    sep    = "|------|-------------------|-----------|-----------|----------------|"
    rows   = []
    for r in records:
        ts   = r.get("timestamp", "—")
        rows.append(
            f"| {ts} | {r.get('irradiance', 0):.1f} "
            f"| {r.get('cloud_cover', 0):.0f} "
            f"| {r.get('temperature', 0):.1f} "
            f"| {r.get('predicted_generation', 0):.2f} |"
        )
    return "\n".join([header, sep] + rows)


# =====================================================================
# 3. NODE: Forecaster_Analyzer
# =====================================================================
def forecaster_analyzer(state: AgentState) -> dict:
    """
    Interprets the ML model's hourly output.
    Produces a concise textual summary of the generation profile.
    """
    records = state["forecast_data"]
    if not records:
        return {"forecast_summary": "No forecast data provided.", "errors": ["Empty forecast_data"]}

    table = _format_forecast_table(records)
    total_kwh = sum(r.get("predicted_generation", 0) for r in records) / 1000
    peak = max(records, key=lambda r: r.get("predicted_generation", 0))
    trough = min(records, key=lambda r: r.get("predicted_generation", 0))

    prompt = (
        f"You are a solar energy analyst. Below is an hourly Linear Regression "
        f"forecast for today's solar generation.\n\n"
        f"{table}\n\n"
        f"Key stats:\n"
        f"- Total estimated energy: {total_kwh:.3f} kWh\n"
        f"- Peak generation: {peak.get('predicted_generation', 0):.2f} W "
        f"at {peak.get('timestamp', '?')}\n"
        f"- Minimum generation: {trough.get('predicted_generation', 0):.2f} W "
        f"at {trough.get('timestamp', '?')}\n\n"
        f"Provide a concise 3–4 sentence narrative analysis of this solar generation "
        f"profile highlighting generation trends, potential morning ramp, midday plateau, "
        f"and evening ramp-down. Be specific with numbers."
    )

    system = "You are a professional solar grid analyst. Respond precisely and concisely."
    summary = _llm_call(system, prompt)
    return {"forecast_summary": summary}


# =====================================================================
# 4. NODE: Risk_Assessor
# =====================================================================
def risk_assessor(state: AgentState) -> dict:
    """
    Identifies periods of high generation variability and risk.
    Flags sudden irradiance drops, high cloud cover, low-generation windows.
    """
    records = state["forecast_data"]
    forecast_summary = state.get("forecast_summary", "")

    # ── Rule-based risk detection ──────────────────────────────────────
    risk_flags = []
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        irr_prev = prev.get("irradiance", 0)
        irr_curr = curr.get("irradiance", 0)
        cloud    = curr.get("cloud_cover", 0)
        gen      = curr.get("predicted_generation", 0)
        ts       = curr.get("timestamp", f"Hour {i}")

        # Sudden irradiance drop > 25%
        if irr_prev > 0 and ((irr_prev - irr_curr) / irr_prev) > 0.25:
            drop_pct = round(((irr_prev - irr_curr) / irr_prev) * 100, 1)
            risk_flags.append(
                f"⚠️  [{ts}] Irradiance drop of {drop_pct}% "
                f"({irr_prev:.0f} → {irr_curr:.0f} W/m²) — HIGH VARIABILITY"
            )

        # Very high cloud cover
        if cloud > 60:
            risk_flags.append(
                f"🌩️  [{ts}] Cloud cover at {cloud:.0f}% — SEVERE OBSTRUCTION"
            )
        elif cloud > 40:
            risk_flags.append(
                f"☁️  [{ts}] Cloud cover at {cloud:.0f}% — MODERATE RISK"
            )

        # Near-zero generation during expected solar hours (08–16)
        try:
            hour = int(str(ts).split(" ")[1].split(":")[0]) if " " in str(ts) else i
        except Exception:
            hour = i
        if 8 <= hour <= 16 and gen < 10:
            risk_flags.append(
                f"🔴 [{ts}] Generation near-zero ({gen:.2f} W) during peak hours — CRITICAL"
            )

    risk_text = "\n".join(risk_flags) if risk_flags else "✅ No significant risks detected."

    prompt = (
        f"You are a grid risk analyst. Given this solar forecast summary and the "
        f"auto-detected risk events, produce a structured risk assessment.\n\n"
        f"FORECAST SUMMARY:\n{forecast_summary}\n\n"
        f"AUTO-DETECTED EVENTS:\n{risk_text}\n\n"
        f"Output a risk assessment with:\n"
        f"1. Overall risk level (LOW / MEDIUM / HIGH / CRITICAL)\n"
        f"2. Top 3 risk periods with timestamps and severity\n"
        f"3. Primary risk drivers\n"
        f"Keep it concise and actionable for a grid operator."
    )

    system = "You are a grid stability and risk analysis expert."
    assessment = _llm_call(system, prompt)
    return {"risk_assessment": assessment}


# =====================================================================
# 5. NODE: RAG_Retriever
# =====================================================================
def rag_retriever(state: AgentState) -> dict:
    """
    Builds a targeted query from the risk assessment and retrieves
    relevant industry protocols from the FAISS vector store.
    """
    risk_text = state.get("risk_assessment", "")
    forecast_summary = state.get("forecast_summary", "")

    # Compose a rich retrieval query
    query = (
        f"Grid management protocols for: {forecast_summary[:300]} "
        f"Risk factors: {risk_text[:300]}"
    )

    try:
        docs = retrieve_docs(query, k=5)
    except Exception as e:
        docs = f"[RAG Error: {e}] — Using fallback: Apply spinning reserves for drops > 25% irradiance."

    return {"retrieved_docs": docs}


# =====================================================================
# 6. NODE: Strategy_Planner
# =====================================================================
def strategy_planner(state: AgentState) -> dict:
    """
    Synthesises forecast, risk, and retrieved protocols into a concrete
    battery / grid utilisation action plan.
    """
    records = state["forecast_data"]
    forecast_summary = state.get("forecast_summary", "")
    risk_assessment  = state.get("risk_assessment", "")
    retrieved_docs   = state.get("retrieved_docs", "")

    # Build a structured prompt
    prompt = (
        f"You are a grid optimization strategist for a solar microgrid.\n\n"
        f"=== FORECAST SUMMARY ===\n{forecast_summary}\n\n"
        f"=== RISK ASSESSMENT ===\n{risk_assessment}\n\n"
        f"=== RETRIEVED INDUSTRY PROTOCOLS ===\n{retrieved_docs}\n\n"
        f"Based on the above, generate a detailed, hour-by-hour ENERGY UTILIZATION PLAN "
        f"covering ALL {len(records)} forecast hours. For each hour specify:\n"
        f"  - Action: CHARGE_BATTERY | DISCHARGE_BATTERY | EXPORT_GRID | IMPORT_GRID | STANDBY\n"
        f"  - Rationale (one line)\n"
        f"  - Priority: HIGH | MEDIUM | LOW\n\n"
        f"Then provide:\n"
        f"  A. Battery Management Summary (charge/discharge schedule)\n"
        f"  B. Grid Interaction Summary (export/import windows)\n"
        f"  C. Demand Response Triggers (if any)\n"
        f"  D. Reserve Activation Needs\n\n"
        f"Format the hour-by-hour plan as a Markdown table. Be specific, actionable, "
        f"and directly reference protocols from the retrieved documents."
    )

    system = (
        "You are an expert energy management system (EMS) planning agent. "
        "Your recommendations must be precise, protocol-compliant, and grid-safe."
    )
    recommendations = _llm_call(system, prompt)
    return {"final_recommendations": recommendations}


# =====================================================================
# 7. NODE: Report_Formatter
# =====================================================================
def report_formatter(state: AgentState) -> dict:
    """
    Assembles all node outputs into a professional Markdown report.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = state["forecast_data"]
    total_kwh = sum(r.get("predicted_generation", 0) for r in records) / 1000
    peak_gen  = max((r.get("predicted_generation", 0) for r in records), default=0)

    report = f"""# ☀️ HeliosCast — Grid Optimization Report
**Generated:** {now}  
**Model:** Linear Regression | **Forecast Horizon:** {len(records)} hours

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total Forecasted Energy | **{total_kwh:.3f} kWh** |
| Peak Generation | **{peak_gen:.2f} W** |
| Forecast Hours | **{len(records)}** |
| Report Type | Grid Optimization with Agentic RAG |

---

## 2. Forecast Analysis

{state.get("forecast_summary", "_Not available_")}

---

## 3. Risk Assessment

{state.get("risk_assessment", "_Not available_")}

---

## 4. Retrieved Industry Protocols

<details>
<summary>📚 Click to expand retrieved protocol excerpts</summary>

{state.get("retrieved_docs", "_Not available_")}

</details>

---

## 5. Grid Optimization Strategy & Recommendations

{state.get("final_recommendations", "_Not available_")}

---

## 6. Appendix — Raw Forecast Data

| Timestamp | Irradiance (W/m²) | Cloud (%) | Temp (°C) | Generation (W) |
|-----------|-------------------|-----------|-----------|----------------|
"""

    for r in records:
        report += (
            f"| {r.get('timestamp', '—')} "
            f"| {r.get('irradiance', 0):.1f} "
            f"| {r.get('cloud_cover', 0):.0f} "
            f"| {r.get('temperature', 0):.1f} "
            f"| {r.get('predicted_generation', 0):.2f} |\n"
        )

    report += f"\n---\n*Report generated by HeliosCast Grid Optimization Agent — LangGraph v0.2*\n"

    return {"report": report}


# =====================================================================
# 8. GRAPH CONSTRUCTION
# =====================================================================
def build_grid_agent_graph() -> StateGraph:
    """Compiles and returns the LangGraph StateGraph."""

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("Forecaster_Analyzer", forecaster_analyzer)
    graph.add_node("Risk_Assessor",       risk_assessor)
    graph.add_node("RAG_Retriever",       rag_retriever)
    graph.add_node("Strategy_Planner",    strategy_planner)
    graph.add_node("Report_Formatter",    report_formatter)

    # Define linear pipeline edges
    graph.set_entry_point("Forecaster_Analyzer")
    graph.add_edge("Forecaster_Analyzer", "Risk_Assessor")
    graph.add_edge("Risk_Assessor",       "RAG_Retriever")
    graph.add_edge("RAG_Retriever",       "Strategy_Planner")
    graph.add_edge("Strategy_Planner",    "Report_Formatter")
    graph.add_edge("Report_Formatter",    END)

    return graph.compile()


# =====================================================================
# 9. PUBLIC ENTRYPOINT
# =====================================================================
def run_grid_agent(forecast_records: List[Dict[str, Any]]) -> AgentState:
    """
    Runs the full LangGraph workflow given a list of forecast records.

    Args:
        forecast_records: List of dicts with keys:
            timestamp, irradiance, cloud_cover, temperature,
            predicted_generation

    Returns:
        The final AgentState dict (all fields populated).
    """
    initial_state: AgentState = {
        "forecast_data":        forecast_records,
        "forecast_summary":     "",
        "risk_assessment":      "",
        "retrieved_docs":       "",
        "final_recommendations":"",
        "report":               "",
        "errors":               [],
    }

    app = build_grid_agent_graph()
    final_state = app.invoke(initial_state)
    return final_state


# =====================================================================
# 10. STANDALONE TEST
# =====================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sample_forecast = [
        {"timestamp": "2026-03-15 06:00", "irradiance": 50.5,  "cloud_cover": 10, "temperature": 18.2, "predicted_generation": 0.0},
        {"timestamp": "2026-03-15 07:00", "irradiance": 150.2, "cloud_cover":  5, "temperature": 20.5, "predicted_generation": 5.2},
        {"timestamp": "2026-03-15 08:00", "irradiance": 350.8, "cloud_cover":  0, "temperature": 22.1, "predicted_generation": 15.8},
        {"timestamp": "2026-03-15 09:00", "irradiance": 550.0, "cloud_cover":  0, "temperature": 24.5, "predicted_generation": 45.3},
        {"timestamp": "2026-03-15 10:00", "irradiance": 750.4, "cloud_cover": 15, "temperature": 26.8, "predicted_generation": 75.1},
        {"timestamp": "2026-03-15 11:00", "irradiance": 850.1, "cloud_cover": 20, "temperature": 28.2, "predicted_generation": 110.5},
        {"timestamp": "2026-03-15 12:00", "irradiance": 920.5, "cloud_cover": 10, "temperature": 29.5, "predicted_generation": 135.2},
        {"timestamp": "2026-03-15 13:00", "irradiance": 880.2, "cloud_cover": 30, "temperature": 30.1, "predicted_generation": 145.8},
        {"timestamp": "2026-03-15 14:00", "irradiance": 720.6, "cloud_cover": 40, "temperature": 29.8, "predicted_generation": 130.4},
        {"timestamp": "2026-03-15 15:00", "irradiance": 510.3, "cloud_cover": 50, "temperature": 28.5, "predicted_generation": 95.1},
        {"timestamp": "2026-03-15 16:00", "irradiance": 280.9, "cloud_cover": 20, "temperature": 26.2, "predicted_generation": 60.8},
        {"timestamp": "2026-03-15 17:00", "irradiance":  95.4, "cloud_cover": 10, "temperature": 24.1, "predicted_generation": 25.4},
        {"timestamp": "2026-03-15 18:00", "irradiance":  10.2, "cloud_cover":  5, "temperature": 22.5, "predicted_generation":  5.1},
    ]

    result = run_grid_agent(sample_forecast)
    print("\n" + "="*80)
    print(result["report"])
