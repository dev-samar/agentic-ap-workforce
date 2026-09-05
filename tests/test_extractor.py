from nodes.extractor import invoice_extraction_node


def test_invoice_extraction_node_without_api_key_uses_text_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    state = {
        "mime_type": "text/html",
        "raw_payload_bytes": (
            b"Invoice #INV-2045\n"
            b"Vendor: Contoso Ltd\n"
            b"Date: 2026-09-05\n"
            b"Due: 2026-09-20\n"
            b"Subtotal: 100.00\n"
            b"Tax: 10.00\n"
            b"Total: 110.00"
        ),
        "pipeline_logs": [],
    }

    result = invoice_extraction_node(state)

    assert result["is_valid_invoice"] is True
    assert result["extracted_data"]["invoice_number"] == "INV-2045"
    assert result["extracted_data"]["vendor_name"] == "Contoso Ltd"
    assert result["extracted_data"]["total_amount_due"] == 110.0
    assert "fallback" in " ".join(result["pipeline_logs"]).lower()
