from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class InvoiceLineItem(BaseModel):
    """Line-item schema designed to train models for high-granularity ERP validation."""
    description: str = Field(description="The description of the specific line-item product or service.")
    quantity: float = Field(default=1.0)
    unit_price: float = Field(default=0.0, description="Cost per unit item.")
    total_price: float = Field(default=0.0, description="Calculated total line value.")

class ExtractionPayload(BaseModel):
    """The master schema extraction target mapped from unstructured formats."""
    vendor_name: Optional[str] = None
    vendor_tax_id: Optional[str] = None  # Crucial for Enterprise Master Data matching
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: str = "INR"
    subtotal: float = 0.0
    tax_amount: float = 0.0
    total_amount_due: float = 0.0
    line_items: List[InvoiceLineItem] = []

class InvoiceState(BaseModel):
    """
    The Master Session State object for LangGraph. 
    Maintains complete context across the middleware pipeline.
    """
    # 1. Routing & Session Variables (Dialogflow-style Session Tracking)
    user_id: str
    tenant_id: str = "retail_default"  # Switches to client corporate IDs for enterprise tenants
    message_id: str
    mime_type: str = "text/html"        # Toggles pipeline fork logic: HTML text vs Scanned PDF
    
    # 2. Binary and Document Payloads
    raw_payload_bytes: Optional[bytes] = None
    optimized_image_path: Optional[str] = None  # Mapped after the OpenCV repair node executes
    
    # 3. Processing Results Gate parameters
    extracted_data: ExtractionPayload = Field(default_factory=ExtractionPayload)
    extraction_confidence: float = 0.0          # The critical gate condition score (Target: >= 0.95)
    is_valid_invoice: bool = False
    pipeline_logs: List[str] = []
    routing_error: Optional[str] = None
