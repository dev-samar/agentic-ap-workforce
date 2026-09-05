from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

# 1. Import your custom middleware configuration modules
from state import InvoiceState
from nodes.ingestion import email_ingestion_node
from nodes.cv_repair import cv_repair_node
from nodes.extractor import invoice_extraction_node

# ==========================================
# CONDITIONAL ROUTER (Dialogflow Transition Rule)
# ==========================================
def ingestion_router_edge(state: Dict[str, Any]) -> Literal["execute_cv_repair", "execute_ai_extractor", "abort_pipeline"]:
    """
    Acts like a DFCX transition route group. Evaluates parameters inside 
    the current session state to determine the next runtime execution page.
    """
    is_valid = state.get("is_valid_invoice", False)
    mime_type = state.get("mime_type", "text/html")
    routing_error = state.get("routing_error")
    
    # Branch A: The stream is corrupted or non-invoice data (Spam filter block)
    if not is_valid or routing_error:
        print(f"[Conditional Edge] Aborting processing flow. Reason: {routing_error}")
        return "abort_pipeline"
        
    # Branch B: Scanned/Printed PDF found. Fork to the OpenCV pre-processing sandbox.
    if mime_type == "application/pdf":
        print("[Conditional Edge] Routing to pure Python OpenCV image correction node.")
        return "execute_cv_repair"
        
    # Branch C: Clean digital inline HTML text e-bill. Route directly to Gemini API.
    print("[Conditional Edge] Routing straight to structured Gemini vision/text node.")
    return "execute_ai_extractor"

def evaluation_gate_edge(state: Dict[str, Any]) -> Literal["commit_to_ledger", "route_to_hitl_exception"]:
    """
    Evaluates confidence parameters returned by the AI node.
    Enforces your premium threshold gate guardrails.
    """
    confidence = state.get("extraction_confidence", 0.0)
    
    # If the score dips below 95%, halt automation and save context parameter layers
    if confidence < 0.95:
        print(f"[Conditional Edge] Confidence score ({confidence}) below 95% threshold target.")
        return "route_to_hitl_exception"
        
    print(f"[Conditional Edge] Confidence verification high ({confidence}). Success.")
    return "commit_to_ledger"

# ==========================================
# BUILDING THE STATEFUL MIDDLEWARE WORKFLOW
# ==========================================

# Initialize the state graph canvas matching your InvoiceState Pydantic contract model properties
workflow = StateGraph(dict) # Initializing with dict for seamless runtime testing cell evaluations

# Step 1: Add your processing Nodes to the canvas topology map
workflow.add_node("ingestion_node", email_ingestion_node)
workflow.add_node("cv_repair_node", cv_repair_node)
workflow.add_node("ai_extractor_node", invoice_extraction_node)

# Step 2: Establish the mandatory execution entry point (The Start Page Trigger)
workflow.set_entry_point("ingestion_node")

# Step 3: Wire up the Conditional Routing Logic from the Ingestion Page
workflow.add_conditional_edges(
    "ingestion_node",
    ingestion_router_edge,
    {
        "execute_cv_repair": "cv_repair_node",
        "execute_ai_extractor": "ai_extractor_node",
        "abort_pipeline": END
    }
)

# Step 4: Map the loop transition out of image repair directly into the extractor node
workflow.add_edge("cv_repair_node", "ai_extractor_node")

# Step 5: Establish the Verification Gate transition exiting your extractor page
# (This switches paths between your B2B dashboard exception screens and downstream automated SAP scripts later)
workflow.add_conditional_edges(
    "ai_extractor_node",
    evaluation_gate_edge,
    {
        "commit_to_ledger": END,
        "route_to_hitl_exception": END  # Both end gracefully for now until external API schemas wire up
    }
)

# Step 6: Compile the state nodes and transitions into an executable machine pipeline
app_engine = workflow.compile()
print("🎉 Success! LangGraph core orchestration topology successfully compiled.")
