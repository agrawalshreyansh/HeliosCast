---
title: HeliosCast
emoji: ☀️
colorFrom: pink
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
short_description: Multi-Model Regression Framework & Grid Optimization Agent
---

# ☀️ HeliosCast: Solar Generation Forecasting & Grid Optimization

HeliosCast is a comprehensive tool for predicting solar energy generation and optimizing grid utilization. It combines a robust Machine Learning forecasting pipeline with an advanced Agentic reasoning engine.

## 🌟 Key Features

1.  **🔮 Real-Time Forecast**: Instantly predict power output based on current or planned weather conditions (Irradiance, Temperature, Cloud Cover) using a trained Linear Regression model.
2.  **📂 Batch Processing**: Upload bulk weather forecast CSVs to generate predictions over extended periods.
3.  **📊 Training & Model Metrics**: View feature importance, error distribution (MAE), and actual vs. predicted curves validating the core ML model.
4.  **🤖 Grid Optimization Agent (LangGraph)**: An advanced 5-node agent pipeline that analyzes forecasts, assesses variability risks, retrieves industry protocols via RAG, and outputs hour-by-hour action plans (Battery/Grid/Reserves).
5.  **💬 Report Chatbot**: Discuss the agent's optimization report with a context-aware chatbot (powered by Gemini Flash) directly in the app.

## 📂 Project Structure

```text
HeliosCast/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── agent/                      # LangGraph Agent & RAG implementation
│   ├── grid_agent.py           # Core 5-node graph logic
│   ├── rag_setup.py            # FAISS vector store builder and retriever
│   ├── faiss_index/            # Compiled vector store (generated)
│   └── knowledge_base/         # Industry protocols (.txt files) for RAG
├── assets/                     # Static resources
│   ├── images/                 # Charts and architecture diagrams
│   └── data/                   # Sample datasets (e.g., test.csv)
├── models/                     # Saved Machine Learning models
│   ├── solar_model.pkl         # Trained predictive model
│   └── scaler.pkl              # Feature scaling object
└── docs/                       # Detailed documentation
    ├── agent_workflow.md
    └── optimization_report_leaflet.md
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   A [Google Gemini API Key](https://aistudio.google.com/) (Free tier is sufficient).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/HeliosCast.git
    cd HeliosCast
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize the RAG Vector Store:**
    *Note: The agent requires the FAISS index to be built locally from the `knowledge_base/` folder before its first run.*
    ```bash
    python agent/rag_setup.py
    ```

### Running the App

Start the Streamlit dashboard:
```bash
streamlit run app.py
```
*   The application will be available at `http://localhost:8501`.
*   Navigate to the **🤖 Grid Optimization Agent** or **💬 Report Chatbot** tabs and enter your Gemini API Key in the sidebar to enable AI features.

## 🧠 The Grid Optimization Agent (LangGraph)

The agent runs an automated pipeline analyzing your generated forecasts:
1.  **Forecaster_Analyzer**: Interprets the ML hourly output into a narrative summary.
2.  **Risk_Assessor**: Identifies periods of high variability (e.g., sudden irradiance drops, heavy cloud cover).
3.  **RAG_Retriever**: Queries the local FAISS store for relevant grid management protocols based on identified risks.
4.  **Strategy_Planner**: Synthesizes a concrete, hour-by-hour energy utilization plan.
5.  **Report_Formatter**: Assembles findings into a downloadable Markdown report.

---
*Built for the End-Semester Delivery Milestone.*
