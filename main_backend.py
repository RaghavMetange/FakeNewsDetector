from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import shutil
import os
import json

from metadata_analyzer import analyze_url_metadata
from ocr_pipeline import run_ocr_pipeline
from intelligence import check_domain_safety, analyze_text_sensationalism

app = FastAPI(title="Fully Unified Multilingual Fake News Detection System")

def load_whitelist():
    try:
        with open("domains_db.json", "r") as f:
            db = json.load(f)
            return db.get("trusted", [])
    except Exception:
        return []

@app.post("/analyze")
async def analyze_input(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    response = {
        "input_types": [],
        "metadata_result": None,
        "ocr_result": None,
        "text_result": None,
        "combined_risk_score": 0.0,
        "flags": []
    }
    
    whitelist = load_whitelist()
    scores = []
    
    # 1. URL & Metadata Engine
    if url:
        response["input_types"].append("url")
        meta_data = analyze_url_metadata(url)
        
        domain = meta_data.get("domain", "")
        domain_risk, domain_flags = check_domain_safety(domain, whitelist)
        
        extracted_text = meta_data.get("extracted_data", {}).get("text_snippet", "") or ""
        text_risk, text_flags = analyze_text_sensationalism(extracted_text)
        
        base_meta_score = meta_data.get("metadata_score", 0.0)
        final_meta_score = min(1.0, base_meta_score + domain_risk + text_risk)
        meta_data["metadata_score"] = round(final_meta_score, 2)
        
        all_meta_flags = list(set(meta_data.get("flags", []) + domain_flags + text_flags))
        meta_data["flags"] = all_meta_flags
        
        response["metadata_result"] = meta_data
        scores.append(meta_data["metadata_score"])
        response["flags"].extend(all_meta_flags)
        
    # 2. Image & OCR Engine
    if file:
        response["input_types"].append("image")
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        ocr_data = run_ocr_pipeline(temp_file_path)
        
        ocr_text = ocr_data.get("extracted_text", "")
        ocr_text_risk, ocr_text_flags = analyze_text_sensationalism(ocr_text)
        
        base_ocr_risk = 0.2 if ocr_data.get("ocr_confidence", 0) > 0.4 else 0.5
        final_ocr_risk = min(1.0, base_ocr_risk + ocr_text_risk)
        
        ocr_data["ocr_risk_score"] = round(final_ocr_risk, 2)
        all_ocr_flags = list(set(ocr_data.get("flags", []) + ocr_text_flags))
        ocr_data["flags"] = all_ocr_flags
        
        response["ocr_result"] = ocr_data
        scores.append(ocr_data["ocr_risk_score"])
        response["flags"].extend(all_ocr_flags)
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    # 3. Direct Text Intelligence Engine (Your Engine)
    if text:
        response["input_types"].append("text")
        text_risk, text_flags = analyze_text_sensationalism(text)
        
        text_result = {
            "analyzed_text": text,
            "text_risk_score": round(text_risk, 2),
            "flags": text_flags
        }
        response["text_result"] = text_result
        scores.append(text_risk)
        response["flags"].extend(text_flags)
        
    # Combine scores across all active inputs
    if scores:
        response["combined_risk_score"] = round(sum(scores) / len(scores), 2)
        
    response["flags"] = list(set(response["flags"]))
    return response