import os
from typing import Dict, Any
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenAI
from state import ExtractionPayload # Our strict session validation schema contract

def invoice_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateful AI Node: Passes the pre-processed visual asset path or text body 
    to Gemini 1.5 Flash using strict structured output constraints.
    """
    print("\n[Node ── AI Extractor] Initiating schema generation parsing...")
    
    logs = state.get("pipeline_logs", [])
    mime_type = state.get("mime_type", "text/html")
    
    # 1. Initialize our cost-effective Gemini connection client
    # Looks for GEMINI_API_KEY environment configuration variable
    try:
        llm = ChatGoogleGenAI(model="gemini-1.5-flash", temperature=0.0)
        # Bind the model directly to our state validation Pydantic contract definition
        structured_llm = llm.with_structured_output(ExtractionPayload)
    except Exception as e:
        return {"routing_error": f"LLM client setup failed: {str(e)}", "is_valid_invoice": False}

    try:
        # 2. Branch A: Process an optimized image path asset (Printed/Scanned Corporate Track)
        if mime_type == "application/pdf" or state.get("optimized_image_path"):
            image_target_path = state.get("optimized_image_path")
            if not image_target_path or not os.path.exists(image_target_path):
                return {"routing_error": "Optimized image asset missing for vision node path.", "is_valid_invoice": False}
            
            logs.append("Constructing multi-modal prompt payload containing optimized image array frame.")
            
            # Format multi-modal context parameter lists for LangChain structure
            prompt_payload = [
                {
                    "type": "text",
                    "text": "Analyze this cleaned corporate invoice image scan. Extract all schema metrics faithfully. If a field is obscure or unreadable, return null."
                },
                {
                    "type": "image_url",
                    "image_url": image_target_path # LangChain automatically resolves local path frames into media streams
                }
            ]
        
        # 3. Branch B: Process standard text/HTML blocks (Crisp Retail E-Bill Track)
        else:
            raw_text = state.get("raw_payload_bytes", b"").decode('utf-8', errors='ignore')
            logs.append("Constructing text-based semantic extraction prompt payload.")
            
            prompt_payload = f"""
            Analyze the following transactional text payload body extracted from an email. 
            Isolate semantic variables and match them exactly to the output data schema parameters.
            
            Raw Ingested Data Stream:
            -------------------------
            {raw_text}
            """

        # 4. Invoke the AI model. LangChain forces output matching our Pydantic properties.
        parsed_results = structured_llm.invoke(prompt_payload)
        logs.append("Frontier schema parsing complete. Structuring output dictionary parameters.")
        
        # Mock calculation metric for testing gate thresholds (will scale via system confidence evaluation checks)
        calculated_confidence = 0.98 
        
        return {
            "extracted_data": parsed_results.model_dump(),
            "extraction_confidence": calculated_confidence,
            "is_valid_invoice": True,
            "pipeline_logs": logs + ["Extraction data state metrics written completely."]
        }
        
    except Exception as e:
        print(f"[AI Extraction Failure] Loop exception caught: {str(e)}")
        return {
            "routing_error": f"AI Parsing pipeline failure: {str(e)}",
            "is_valid_invoice": False
        }
