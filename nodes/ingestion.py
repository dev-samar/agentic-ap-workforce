import email
from email import policy
from typing import Dict, Any
import pydantic

def email_ingestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure Python Node: Simulates a Dialogflow CX entry trigger.
    Receives raw email payload bytes from the master session state, 
    parses headers dynamically, and filters for text or attachment streams.
    """
    print(f"\n[Node ── Ingestion] Initializing payload extraction for Message ID: {state.get('message_id')}")
    
    # 1. Access the master session state inputs
    raw_bytes = state.get("raw_payload_bytes")
    if not raw_bytes:
        print("[Ingestion Error] Received empty or Null byte stream.")
        return {
            "routing_error": "Payload payload_bytes missing at runtime.",
            "is_valid_invoice": False
        }
    
    # 2. Parse raw bytes natively using standard library with strict SMTP/MIME policy
    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    except Exception as e:
        return {
            "routing_error": f"MIME parser failure: {str(e)}",
            "is_valid_invoice": False
        }
        
    target_content = None
    detected_mime = "text/html"
    logs = ["Email byte structure successfully decoded."]
    
    # 3. Handle Multi-part messages (Emails with inline bodies + physical attachments)
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Branch A: Scanned/Printed PDF Attachment (The messy corporate layout track)
            if "attachment" in content_disposition and "pdf" in content_type.lower():
                logs.append("Physical PDF invoice attachment isolated in MIME layout.")
                print("[Ingestion] Branch Detected: Scanned / Printed PDF document found.")
                return {
                    "raw_payload_bytes": part.get_content(),
                    "mime_type": "application/pdf",
                    "is_valid_invoice": True,
                    "pipeline_logs": state.get("pipeline_logs", []) + logs
                }
            
            # Branch B: Standard HTML Email body (The crisp retail e-bill track)
            elif content_type == "text/html" and not target_content:
                target_content = part.get_content()
    else:
        # Fallback if email is a flat text stream without structural headers
        if msg.get_content_type() == "text/html":
            target_content = msg.get_content()
        else:
            # Handle plain text format fallbacks
            target_content = msg.get_content()
            detected_mime = "text/plain"

    if not target_content:
        print("[Ingestion Warning] Node failed to isolate clear body text or PDF arrays.")
        return {
            "routing_error": "Unable to extract readable invoice layer from stream structures.",
            "is_valid_invoice": False
        }
        
    logs.append(f"Inline transaction text isolated. Formatted context: {detected_mime}")
    print(f"[Ingestion] Branch Detected: Inline digital text structure found ({detected_mime}).")
    
    # 4. Return updates to merge cleanly with our state parameters dictionary
    return {
        "raw_payload_bytes": target_content.encode('utf-8') if isinstance(target_content, str) else target_content,
        "mime_type": detected_mime,
        "is_valid_invoice": True,
        "pipeline_logs": state.get("pipeline_logs", []) + logs
    }
