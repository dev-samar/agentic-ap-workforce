import os
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from state import ExtractionPayload


def _coerce_float(value: Optional[str]) -> float:
    if value is None:
        return 0.0
    cleaned = value.replace(",", "").replace("₹", "").replace("$", "").replace("€", "").replace("£", "").strip()
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return 0.0


def _extract_invoice_fallback(raw_text: str) -> Dict[str, Any]:
    """Deterministic parser used when Gemini is unavailable or fails."""
    payload: Dict[str, Any] = {
        "vendor_name": None,
        "vendor_tax_id": None,
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "currency": "INR",
        "subtotal": 0.0,
        "tax_amount": 0.0,
        "total_amount_due": 0.0,
        "line_items": [],
    }

    if not raw_text:
        return payload

    text = raw_text.replace("\r", " ").replace("\n", " ")
    text_for_matches = raw_text

    vendor_match = re.search(r"Vendor\s*[:\-]?\s*([A-Za-z0-9 .&'-]+)", text_for_matches, re.IGNORECASE)
    if vendor_match:
        payload["vendor_name"] = vendor_match.group(1).strip()

    invoice_match = re.search(r"(?:invoice(?:\s*(?:no|#|number))?|inv(?:oice)?)\s*[:#-]?\s*([A-Z0-9-]+)", text_for_matches, re.IGNORECASE)
    if invoice_match:
        payload["invoice_number"] = invoice_match.group(1).strip()

    for label in ("invoice date", "date"):
        date_match = re.search(rf"{label}\s*[:\-]?\s*(\d{{4}}-\d{{2}}-\d{{2}}|\d{{2}}/\d{{2}}/\d{{4}}|\d{{4}}/\d{{2}}/\d{{2}})", text_for_matches, re.IGNORECASE)
        if date_match:
            payload["invoice_date"] = date_match.group(1)
            break

    due_match = re.search(r"(?:due\s*date|payment due)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2})", text_for_matches, re.IGNORECASE)
    if due_match:
        payload["due_date"] = due_match.group(1)

    currency_match = re.search(r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD)\b", text_for_matches, re.IGNORECASE)
    if currency_match:
        payload["currency"] = currency_match.group(1).upper()

    subtotal_match = re.search(r"\b(?:subtotal|sub\s*total|amount\s+before\s+tax)\b\s*[:\-]?\s*[$₹€£]?\s*([0-9,]+(?:\.\d{1,2})?)", text_for_matches, re.IGNORECASE)
    if subtotal_match:
        payload["subtotal"] = _coerce_float(subtotal_match.group(1))

    tax_match = re.search(r"\b(?:tax|gst|vat|sales\s+tax)\b\s*[:\-]?\s*[$₹€£]?\s*([0-9,]+(?:\.\d{1,2})?)", text_for_matches, re.IGNORECASE)
    if tax_match:
        payload["tax_amount"] = _coerce_float(tax_match.group(1))

    total_match = re.search(r"\b(?:total(?:\s+amount\s+due)?|grand\s+total|amount\s+due)\b\s*[:\-]?\s*[$₹€£]?\s*([0-9,]+(?:\.\d{1,2})?)", text_for_matches, re.IGNORECASE)
    if total_match:
        payload["total_amount_due"] = _coerce_float(total_match.group(1))

    if payload["total_amount_due"] == 0.0 and payload["subtotal"] and payload["tax_amount"]:
        payload["total_amount_due"] = payload["subtotal"] + payload["tax_amount"]

    return payload


def invoice_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateful AI Node: Passes the pre-processed visual asset path or text body
    to Gemini 1.5 Flash using strict structured output constraints.
    """
    print("\n[Node ── AI Extractor] Initiating schema generation parsing...")

    logs = list(state.get("pipeline_logs", []))
    mime_type = state.get("mime_type", "text/html")

    raw_text = ""
    if mime_type == "application/pdf" or state.get("optimized_image_path"):
        image_target_path = state.get("optimized_image_path")
        if not image_target_path or not os.path.exists(image_target_path):
            return {"routing_error": "Optimized image asset missing for vision node path.", "is_valid_invoice": False}
        logs.append("Constructing multi-modal prompt payload containing optimized image array frame.")
    else:
        raw_payload = state.get("raw_payload_bytes", b"")
        if isinstance(raw_payload, bytes):
            raw_text = raw_payload.decode("utf-8", errors="ignore")
        elif isinstance(raw_payload, str):
            raw_text = raw_payload
        logs.append("Constructing text-based semantic extraction prompt payload.")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logs.append("No Gemini API key detected; using deterministic local fallback extraction.")
        fallback_payload = _extract_invoice_fallback(raw_text)
        is_valid = bool(fallback_payload.get("invoice_number") or fallback_payload.get("total_amount_due") or fallback_payload.get("vendor_name"))
        return {
            "extracted_data": fallback_payload,
            "extraction_confidence": 0.82,
            "is_valid_invoice": is_valid,
            "pipeline_logs": logs + ["Local fallback extraction completed successfully."],
        }

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, api_key=api_key)
        structured_llm = llm.with_structured_output(ExtractionPayload)

        if mime_type == "application/pdf" or state.get("optimized_image_path"):
            image_target_path = state.get("optimized_image_path")
            prompt_payload = [
                {
                    "type": "text",
                    "text": "Analyze this cleaned corporate invoice image scan. Extract all schema metrics faithfully. If a field is obscure or unreadable, return null.",
                },
                {
                    "type": "image_url",
                    "image_url": image_target_path,
                },
            ]
        else:
            prompt_payload = f"""
            Analyze the following transactional text payload body extracted from an email.
            Isolate semantic variables and match them exactly to the output data schema parameters.

            Raw Ingested Data Stream:
            -------------------------
            {raw_text}
            """

        parsed_results = structured_llm.invoke(prompt_payload)
        logs.append("Frontier schema parsing complete. Structuring output dictionary parameters.")

        calculated_confidence = 0.98

        return {
            "extracted_data": parsed_results.model_dump(),
            "extraction_confidence": calculated_confidence,
            "is_valid_invoice": True,
            "pipeline_logs": logs + ["Extraction data state metrics written completely."],
        }

    except Exception as e:
        print(f"[AI Extraction Failure] Loop exception caught: {str(e)}")
        if raw_text:
            logs.append(f"LLM failed, using local parser fallback instead: {str(e)}")
            fallback_payload = _extract_invoice_fallback(raw_text)
            is_valid = bool(fallback_payload.get("invoice_number") or fallback_payload.get("total_amount_due") or fallback_payload.get("vendor_name"))
            return {
                "extracted_data": fallback_payload,
                "extraction_confidence": 0.82,
                "is_valid_invoice": is_valid,
                "pipeline_logs": logs + ["Fallback extraction completed after AI failure."],
            }
        return {
            "routing_error": f"AI Parsing pipeline failure: {str(e)}",
            "is_valid_invoice": False,
        }
