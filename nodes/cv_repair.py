import cv2
import numpy as np
import os
from typing import Dict, Any

def cv_repair_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure Python Node: Evaluates printed scans or binary attachments.
    Executes automated deskewing (straightening) and adaptive thresholding
    to flatten geometric distortions and isolate crisp black-and-white text lines.
    """
    print("\n[Node ── CV Repair] Processing physical layout alignment matrix...")
    
    raw_bytes = state.get("raw_payload_bytes")
    message_id = state.get("message_id", "unknown_msg")
    logs = state.get("pipeline_logs", [])
    
    if not raw_bytes:
        print("[CV Repair Warning] No byte stream array found in memory state.")
        return {"routing_error": "CV Repair triggered with empty payload.", "is_valid_invoice": False}

    try:
        # 1. Decode raw file bytes into an OpenCV grayscale image array matrix
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # If byte decoding fails directly, fallback to routing exception check
            return {"routing_error": "Failed to decode binary array stream into image matrix.", "is_valid_invoice": False}
        
        logs.append("Binary payload array decoded into grayscale image layer.")

        # 2. Deskewing Script (Automated alignment correction)
        # Invert the pixel values: text maps to white (255), canvas background to black (0)
        thresh_invert = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Grab all non-zero pixel coordinates to map text structural bounding bounds
        coords = np.column_stack(np.where(thresh_invert > 0))
        
        # Calculate the minimal area box encompassing text features to compute angle slant
        angle = cv2.minAreaRect(coords)[-1]
        
        # Standardize angle layout quadrants from raw output structures
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate image array back to perfect horizontal grid coordinates
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        straightened_img = cv2.warpAffine(
            img, rotation_matrix, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=255
        )
        logs.append(f"Document rotation corrected by deskewing matrix: {round(angle, 2)} degrees.")

        # 3. High-Contrast Adaptive Thresholding (Erases stamps, signatures, and shadows)
        clean_binary_img = cv2.adaptiveThreshold(
            straightened_img, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15,  # Pixel pixel evaluating neighborhood window block matrix size
            9    # Contrast clip offset boundary constant
        )
        logs.append("Adaptive threshold filters applied. Extracted clean text grid outlines.")

        # 4. Save optimized file asset path into our data samples folder directory
        output_dir = "data/samples"
        os.makedirs(output_dir, exist_ok=True)
        optimized_file_path = os.path.join(output_dir, f"clean_{message_id}.png")
        cv2.imwrite(optimized_file_path, clean_binary_img)
        
        print(f"[CV Repair] Success! Optimized text line asset written to: {optimized_file_path}")
        
        return {
            "optimized_image_path": optimized_file_path,
            "pipeline_logs": logs + [f"Cleaned multi-modal asset stored locally at: {optimized_file_path}"]
        }
        
    except Exception as e:
        print(f"[CV Repair Failure] Programmatic calculation error: {str(e)}")
        return {
            "routing_error": f"Image processing failed: {str(e)}",
            "is_valid_invoice": False
        }
