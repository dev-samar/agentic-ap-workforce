# agentic-ap-workforce
Autonomous multi-tenant agentic ai workforce for retail email parsing and enterprise AP automation
# Agentic AP Workforce — Core Middleware Engine

An autonomous, multi-tenant Agentic AI Workforce middleware platform designed to orchestrate complex Accounts Payable (AP) automation pipelines. Built natively using a deterministic graph network skeleton, the platform seamlessly supports both retail B2C consumer email ingestion and enterprise B2B on-premise SAP R/3 (ECC) transaction data entry.

## 🏛️ System Architecture & Data Flow

The core engine is driven by **LangGraph**, treating the data processing path as a stateful, cyclic state machine rather than a black-box chatbot. Rather than relying on fragile LLM tool-calling loops, the pipeline forces absolute programmatic control by interweaving pure Python algorithmic nodes with advanced vision-capable Small Language Models (SLMs).

[ INBOUND STREAM ]Multipart Email Bytes, Scanned PDF, Images│▼┌──────────────────────────┐│    ingestion_node        ││ (Pure Python MIME Split) │└────────────┬─────────────┘│[ CONDITIONAL EDGE ]Fork Pipeline Based on Mime-Type Property│┌───────────────────────┴───────────────────────┐▼ (Mime: application/pdf)                       ▼ (Mime: text/html)┌──────────────────────────┐                            ││     cv_repair_node       │                            ││ (Pure Python OpenCV lines│                            ││  Deskew & Binarization)  │                            │└───────────┬──────────────┘                            ││                                           │└───────────────────┬───────────────────────┘│▼┌──────────────────────────┐│    ai_extractor_node     ││ (Gemini 1.5 Flash Vision ││  Strict Pydantic JSON)   │└────────────┬─────────────┘│[ CONDITIONAL EDGE ]Evaluation Gate Threshold Check (Target >= 95%)│┌───────────────────────┴───────────────────────┐▼ (Confidence >= 0.95)                          ▼ (Confidence < 0.95)┌──────────────────────────┐                    ┌──────────────────────────┐│     [ SAP COMMIT ]       │                    │    [ HITL DASHBOARD ]    ││  Straight-Through BAPI   │                    │ Manual AP Exception Queue│└──────────────────────────┘                    └──────────────────────────┘

## 📂 Core Code Directory Mapping

The repository isolates computational logic nodes completely from global execution runtimes to allow independent testing inside sandboxed cloud testing view ports:

*   `state.py` — The strict **Pydantic V2 Global Session Contract Object**. Maps every tracing lifecycle variable parameter across middleware steps (similar to Dialogflow CX session variables).
*   `graph.py` — The Master Topology Routing Network Canvas. Compiles transition boundaries and evaluation gate routes.
*   `sandbox.ipynb` — Interactive visual testing laboratory for sandboxing OpenCV pixel parameters and inspecting model JSON payload inputs inline.
*   `nodes/` — Isolated computational node package directory.
    *   `nodes/ingestion.py` — Programmatic parsing framework to isolate data bodies or stream raw document attachments.
    *   `nodes/cv_repair.py` — Computer Vision pipeline. Straightens crooked scans (deskewing) and flattens scanner shadow noise using adaptive binarization algorithms.
    *   `nodes/extractor.py` — Structural AI mapping wrapper forced by `ChatGoogleGenerativeAI` to ensure syntax-perfect model schemas without hallucinations.

## 🛠️ Tech Stack & Requirements

Dependencies are frozen inside `requirements.txt` to align strictly with container specifications and keep local runtime dependencies stable:

*   **Orchestration:** LangGraph (StateGraph architecture)
*   **AI Integration:** LangChain Core + LangChain Google GenAI
*   **Computer Vision:** OpenCV Headless (`opencv-python-headless` for serverless cloud compliance)
*   **Data Validation:** Pydantic V2 (`>=2.7.0,<3.0.0` semantic boundaries)
*   **Staging Visualizers:** Matplotlib & Pillow

## 🚀 Running Your First Sandbox Simulation Loop

To test the compiled graph execution states inside your **GitHub Codespace Cloud Sandbox** container:

1. Launch your interactive playground notebook file `sandbox.ipynb`.
2. Configure your live provider access parameter within your terminal layer:
   ```bash
   export GEMINI_API_KEY="your_secret_credential_token"
   ```
3. Run the orchestration verification cell script:
   ```python
   from graph import app_engine
   
   initial_payload = {
       "user_id": "architect_dev_1",
       "tenant_id": "retail_default",
       "message_id": "msg_001_session",
       "raw_payload_bytes": b"From: utility@provider.com\nSubject: Invoice\n\nTotal Due: INR 2499.00",
       "pipeline_logs": []
   }
   
   final_output = app_engine.invoke(initial_payload)
   print(f"Pipeline Flow Success Status: {final_output.get('is_valid_invoice')}")
   ```

## 🔒 Enterprise Guardrails & Compliance Notes
*   **Zero LLM Writing Access:** The AI node only constructs structured metadata. No LLM ever interacts with system databases or pushes direct code into external ERP architectures. 
*   **Deterministic Entry:** Downstream database commits are handled exclusively by pure Python nodes invoking static, cryptographically secured SAP BAPIs or localized network RFC connectors.
💾 Committing the Documentation Asset to GitOnce the file is saved (Ctrl + S or Cmd + S), maximize your integrated bash terminal command pane below and deploy this architectural bluep