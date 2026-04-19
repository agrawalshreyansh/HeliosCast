import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="HeliosCast Pro", page_icon="☀️", layout="wide")

# =====================================================================
# SESSION STATE — initialise ALL keys once at the very top so they
# survive tab switches for the entire browser session.
# =====================================================================
_defaults = {
    "selected_tab":      "🔮 Real-Time Forecast",
    "agent_report":      None,          # last generated Markdown report
    "agent_running":     False,
    "google_api_key":    "",            # persisted API key across tabs
    "chat_history":      [],            # [{role, content}, ...]
    "chat_context_set":  False,         # whether report was injected as context
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Keep os.environ in sync with the stored key (survives re-runs)
if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key

# --- LOAD ASSETS ---
base_path   = os.path.dirname(__file__)
model_path  = os.path.join(base_path, "models", "solar_model.pkl")
scaler_path = os.path.join(base_path, "models", "scaler.pkl")

@st.cache_resource
def load_models():
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_models()
except Exception:
    st.error("Error: Ensure 'solar_model.pkl' and 'scaler.pkl' are in the project folder.")
    st.stop()

feature_names = ["shortwave_radiation", "temperature_2m", "cloud_cover", "hour", "month", "lag_1h"]

# =====================================================================
# SIDEBAR — navigation + persistent API key widget
# =====================================================================
st.sidebar.title("☀️ HeliosCast")
st.sidebar.markdown("---")
st.sidebar.header("Navigation")

TABS = [
    "🔮 Real-Time Forecast",
    "📂 Batch Processing",
    "📊 Training & Model Metrics",
    "🤖 Grid Optimization Agent",
    "💬 Report Chatbot",
]
for tab in TABS:
    if st.sidebar.button(tab, use_container_width=True):
        st.session_state.selected_tab = tab

st.sidebar.markdown("---")

# ── Persistent API key ────────────────────────────────────────────────
with st.sidebar.expander("⚙️ Gemini API Key", expanded=True):
    key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.google_api_key,
        type="password",
        placeholder="AIza...",
        help="Used by both the Grid Agent and the Chatbot. Get one free at https://aistudio.google.com/",
        key="api_key_widget",
    )
    # Persist to session state + env whenever the user types
    if key_input != st.session_state.google_api_key:
        st.session_state.google_api_key = key_input
        os.environ["GOOGLE_API_KEY"] = key_input

    if st.session_state.google_api_key:
        st.success("API key set ✓")
    else:
        st.warning("Enter key to enable agent & chatbot")

# =====================================================================
# HEADER
# =====================================================================
st.title("☀️ HeliosCast: Solar Generation Forecasting")
st.markdown("### Professional Energy Analytics Dashboard")
st.divider()

# =====================================================================
# TAB 1 — Real-Time Forecast
# =====================================================================
if st.session_state.selected_tab == "🔮 Real-Time Forecast":
    st.header("Individual Scenario Simulation")
    col_a, col_b = st.columns(2)
    with col_a:
        irradiance = st.slider("Shortwave Radiation (W/m²)", 0, 1100, 600)
        temp       = st.slider("Temperature (°C)", -10, 50, 28)
        clouds     = st.slider("Cloud Cover (%)", 0, 100, 15)
    with col_b:
        lag_1h  = st.number_input("Last Hour Generation (Watts)", value=120.0)
        date_in = st.date_input("Forecast Date", datetime.now())
        time_in = st.time_input("Forecast Time", datetime.now())

    if st.button("Predict Power Output", use_container_width=True):
        hour, month   = time_in.hour, date_in.month
        features      = np.array([[irradiance, temp, clouds, hour, month, lag_1h]])
        features_sc   = scaler.transform(features)
        prediction    = max(0, round(model.predict(features_sc)[0], 4))
        st.metric(label="Estimated Generation", value=f"{prediction} Watts")
        st.progress(min(1.0, prediction / 1000.0))

# =====================================================================
# TAB 2 — Batch Processing
# =====================================================================
elif st.session_state.selected_tab == "📂 Batch Processing":
    st.header("Bulk CSV Forecasting")
    uploaded_file = st.file_uploader("Upload weather forecast CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"]  = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month
        if all(col in df.columns for col in feature_names):
            X_scaled = scaler.transform(df[feature_names])
            df["predicted_generation"] = model.predict(X_scaled).clip(min=0)
            st.success("Batch Prediction Complete!")
            st.line_chart(df.set_index(df.columns[0])[["predicted_generation"]].head(100))
            st.dataframe(df)
            st.download_button("Download Predictions", df.to_csv(index=False), "helios_results.csv")

# =====================================================================
# TAB 3 — Training & Model Metrics
# =====================================================================
elif st.session_state.selected_tab == "📊 Training & Model Metrics":
    st.header("Model Performance & Training History")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Linear Regression")
    c2.metric("R² Score", "0.8043")
    c3.metric("MAE", "0.05 Watts")
    st.divider()

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("1. Feature Importance")
        p = os.path.join(base_path, "assets", "images", "Feature_Impact.png")
        if os.path.exists(p):
            st.image(p, caption="How different weather factors impact power.")
        else:
            st.warning("Feature_Impact.png not found.")
    with col_img2:
        st.subheader("2. Model Error Distribution")
        p = os.path.join(base_path, "assets", "images", "MEA.png")
        if os.path.exists(p):
            st.image(p, caption="Residual analysis (Errors centered at zero).")
        else:
            st.warning("MEA.png not found.")

    st.subheader("3. Actual vs Predicted Curve")
    p = os.path.join(base_path, "assets", "images", "Prediction.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True, caption="Sample test results showing high correlation.")
    else:
        st.warning("Prediction.png not found.")

    st.subheader("4. Model Comparison")
    p = os.path.join(base_path, "assets", "images", "comparison.png")
    if os.path.exists(p):
        st.image(p, use_column_width=True, caption="Comparison of two models trained on the same data.")
    else:
        st.warning("comparison.png not found.")

# =====================================================================
# TAB 4 — Grid Optimization Agent
# =====================================================================
elif st.session_state.selected_tab == "🤖 Grid Optimization Agent":
    st.header("🤖 LangGraph Grid Optimization Agent")
    st.markdown(
        """
        This agent analyses the ML model's forecast output through a **5-node LangGraph pipeline**:

        `Forecaster_Analyzer` → `Risk_Assessor` → `RAG_Retriever` → `Strategy_Planner` → `Report_Formatter`

        The RAG component retrieves relevant **industry grid protocols** from a local FAISS vector store
        to ground every recommendation in established best practices.
        """
    )
    # ── Workflow Architecture ─────────────────────────────────────────
    st.subheader("Agent Workflow Architecture")
    workflow_path = os.path.join(base_path, "assets", "images", "Helios_Workflow.png")
    if os.path.exists(workflow_path):
        st.image(workflow_path, use_column_width=True, caption="LangGraph 5-Node Agent Pipeline")
    else:
        st.warning("Helios_Workflow.png not found.")

    st.divider()

    # ── Data Source ───────────────────────────────────────────────────
    st.subheader("1. Select Forecast Data Source")
    data_source = st.radio(
        "How would you like to provide forecast data?",
        ["📁 Upload a CSV", "📝 Use built-in test.csv", "🔢 Manually enter records"],
        horizontal=True,
    )

    forecast_records = []

    if data_source == "📁 Upload a CSV":
        agent_csv = st.file_uploader(
            "Upload CSV with columns: timestamp, shortwave_radiation, temperature_2m, cloud_cover, lag_1h",
            type=["csv"], key="agent_csv",
        )
        if agent_csv:
            df_agent = pd.read_csv(agent_csv)
            if "timestamp" in df_agent.columns:
                df_agent["timestamp"] = pd.to_datetime(df_agent["timestamp"])
                df_agent["hour"]  = df_agent["timestamp"].dt.hour
                df_agent["month"] = df_agent["timestamp"].dt.month
            if all(col in df_agent.columns for col in feature_names):
                X_sc = scaler.transform(df_agent[feature_names])
                df_agent["predicted_generation"] = model.predict(X_sc).clip(min=0)
                for _, row in df_agent.iterrows():
                    forecast_records.append({
                        "timestamp":            str(row.get("timestamp", "—")),
                        "irradiance":           float(row.get("shortwave_radiation", 0)),
                        "cloud_cover":          float(row.get("cloud_cover", 0)),
                        "temperature":          float(row.get("temperature_2m", 0)),
                        "predicted_generation": float(row.get("predicted_generation", 0)),
                    })
                st.success(f"✅ Loaded {len(forecast_records)} records. Predictions generated.")
            else:
                st.error(f"CSV must contain columns: {feature_names}")

    elif data_source == "📝 Use built-in test.csv":
        test_path = os.path.join(base_path, "assets", "data", "test.csv")
        if os.path.exists(test_path):
            df_test = pd.read_csv(test_path)
            df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
            df_test["hour"]  = df_test["timestamp"].dt.hour
            df_test["month"] = df_test["timestamp"].dt.month
            X_sc = scaler.transform(df_test[feature_names])
            df_test["predicted_generation"] = model.predict(X_sc).clip(min=0)
            for _, row in df_test.iterrows():
                forecast_records.append({
                    "timestamp":            str(row["timestamp"]),
                    "irradiance":           float(row["shortwave_radiation"]),
                    "cloud_cover":          float(row["cloud_cover"]),
                    "temperature":          float(row["temperature_2m"]),
                    "predicted_generation": float(row["predicted_generation"]),
                })
            st.success(f"✅ Loaded {len(forecast_records)} records from test.csv.")
        else:
            st.error("test.csv not found in project root.")

    else:
        st.info("Enter comma-separated values below (one row per hour).")
        manual_text = st.text_area(
            "Format: timestamp, irradiance, cloud_cover, temperature, lag_1h",
            value="2026-03-15 10:00, 750, 15, 26.8, 75\n2026-03-15 11:00, 850, 20, 28.2, 110",
            height=150,
        )
        for line in manual_text.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    ts, irr, cloud, temp_v, lag = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    hour_v = int(ts.split(" ")[1].split(":")[0]) if " " in ts else 12
                    feat   = np.array([[irr, temp_v, cloud, hour_v, 3, lag]])
                    gen    = float(model.predict(scaler.transform(feat)).clip(min=0)[0])
                    forecast_records.append({
                        "timestamp": ts, "irradiance": irr,
                        "cloud_cover": cloud, "temperature": temp_v,
                        "predicted_generation": gen,
                    })
                except Exception:
                    pass
        if forecast_records:
            st.success(f"✅ Parsed {len(forecast_records)} manual records.")

    # ── Preview ───────────────────────────────────────────────────────
    if forecast_records:
        st.subheader("2. Forecast Preview")
        df_preview = pd.DataFrame(forecast_records)
        st.dataframe(df_preview, use_container_width=True)
        st.line_chart(df_preview.set_index("timestamp")["predicted_generation"])

    # ── Run Agent ─────────────────────────────────────────────────────
    st.subheader("3. Run the Agent")
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button(
            "🚀 Run Grid Agent",
            use_container_width=True,
            disabled=(not forecast_records or not st.session_state.google_api_key),
        )
    with col_info:
        if not st.session_state.google_api_key:
            st.warning("⚠️ Set your Google Gemini API key in the sidebar to enable the agent.")
        elif not forecast_records:
            st.info("ℹ️ Load forecast data above, then click Run.")
        else:
            st.success(f"✅ Ready — {len(forecast_records)} records loaded, API key set.")

    if run_btn and forecast_records:
        try:
            from agent.grid_agent import run_grid_agent
        except ImportError as e:
            st.error(f"Agent dependencies not installed: {e}")
            st.stop()

        with st.status("🤖 Agent pipeline running...", expanded=True) as status:
            st.write("🔍 **Forecaster_Analyzer** — interpreting ML output...")
            st.write("⚠️  **Risk_Assessor** — detecting variability risks...")
            st.write("📚 **RAG_Retriever** — querying FAISS vector store...")
            st.write("📋 **Strategy_Planner** — generating utilisation plan...")
            st.write("📝 **Report_Formatter** — composing final report...")
            try:
                result = run_grid_agent(forecast_records)
                st.session_state.agent_report    = result["report"]
                st.session_state.chat_history    = []   # reset chat when new report arrives
                st.session_state.chat_context_set = False
                status.update(label="✅ Agent pipeline complete!", state="complete")
                st.info("💬 Switch to the **Report Chatbot** tab to ask questions about this report.")
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.exception(e)

    # ── Display Report ────────────────────────────────────────────────
    if st.session_state.agent_report:
        st.divider()
        st.subheader("4. Optimization Report")
        st.markdown(st.session_state.agent_report)
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=st.session_state.agent_report,
            file_name=f"heliocast_grid_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

# =====================================================================
# TAB 5 — Report Chatbot
# =====================================================================
elif st.session_state.selected_tab == "💬 Report Chatbot":
    st.header("💬 Report Chatbot")
    st.markdown(
        "Ask anything about the Grid Optimization Report generated by the agent. "
        "The chatbot is grounded in the report and answers using **Gemini 2.5 Flash**."
    )
    st.divider()

    # ── Guard: need API key ───────────────────────────────────────────
    if not st.session_state.google_api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to use the chatbot.")
        st.stop()

    # ── Guard: need a report ─────────────────────────────────────────
    if not st.session_state.agent_report:
        st.info(
            "📋 No report found yet. Go to **🤖 Grid Optimization Agent**, run the pipeline, "
            "then come back here to chat about the results."
        )
        st.stop()

    # ── Show a compact report preview ────────────────────────────────
    with st.expander("📄 View the current report (context for the chatbot)", expanded=False):
        st.markdown(st.session_state.agent_report)

    st.markdown("---")

    # ── Render existing chat history ─────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about the report… e.g. 'When should I charge the battery?' or 'What is the peak risk period?'"
    )

    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build messages for the LLM
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=st.session_state.google_api_key,
                temperature=0.3,
            )

            # System prompt injects the full report as grounding context
            system_prompt = (
                "You are HeliosCast Assistant, an expert solar energy and grid optimization analyst. "
                "You have been given the following Grid Optimization Report generated by an AI agent. "
                "Answer the user's questions ONLY based on this report. If the answer is not in the report, "
                "say so clearly. Be concise, precise, and reference specific sections or timestamps when relevant.\n\n"
                "=== GRID OPTIMIZATION REPORT ===\n"
                f"{st.session_state.agent_report}\n"
                "=== END OF REPORT ==="
            )

            # Build full message list (system + history)
            lc_messages = [SystemMessage(content=system_prompt)]
            for h in st.session_state.chat_history[:-1]:   # all except the latest user msg
                if h["role"] == "user":
                    lc_messages.append(HumanMessage(content=h["content"]))
                else:
                    lc_messages.append(AIMessage(content=h["content"]))
            lc_messages.append(HumanMessage(content=user_input))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = llm.invoke(lc_messages)
                    answer   = response.content.strip()
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            err_msg = f"❌ Chatbot error: {e}"
            with st.chat_message("assistant"):
                st.error(err_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    # ── Clear chat button ────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()
