import os
import cv2
import numpy as np

# Lazy-loaded OCR reader — avoids ~15s startup penalty when OCR isn't needed
_reader = None

def get_ocr_reader():
    """Initialize EasyOCR reader on first use only."""
    global _reader
    if _reader is None:
        import easyocr
        print("Initializing EasyOCR reader for English, Hindi, and Marathi...")
        _reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=False)
        print("EasyOCR reader initialized successfully!")
    return _reader

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Enhanced preprocessing for complex/dark-mode screenshots:
    1. Loads the image and upscales it 2x (helps OCR read small or compressed text).
    2. Converts to grayscale.
    3. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
       to handle dark backgrounds and uneven lighting/overlays.
    4. Applies Bilateral Filtering to reduce background noise while keeping text edges sharp.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
        
    # 1. Upscale image by 2x using cubic interpolation for higher pixel density
    height, width = img.shape[:2]
    upscaled = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This excels at bringing out text on dark, low-contrast, or gradient backgrounds
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. Bilateral Filter to smooth out noise while preserving sharp character edges
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    return filtered

def run_ocr_pipeline(image_path: str) -> dict:
    """
    Run OCR on an image and return extracted text with confidence.
    OCR is treated as a pure extraction step — credibility assessment 
    is handled downstream by the text intelligence pipeline.
    """
    result = {
        "image_path": image_path,
        "extracted_text": "",
        "ocr_confidence": 0.0,
        "flags": []
    }
    
    try:
        processed_img = preprocess_image(image_path)
        reader = get_ocr_reader()
        
        # Run EasyOCR on the enhanced image
        ocr_results = reader.readtext(processed_img, detail=1)
        
        if not ocr_results:
            result["flags"].append("No text could be detected in the image")
            return result
            
        extracted_texts = []
        confidences = []
        
        for bbox, text, conf in ocr_results:
            # Filter out tiny, low-confidence hallucinated symbols
            if conf > 0.15: 
                extracted_texts.append(text)
                confidences.append(conf)
            
        full_text = " ".join(extracted_texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        result["extracted_text"] = full_text
        result["ocr_confidence"] = round(avg_confidence, 2)
        
        if avg_confidence < 0.4:
            result["flags"].append("Low OCR confidence: screenshot may be blurry or unreadable")
            
    except Exception as e:
        result["flags"].append(f"OCR processing failed: {str(e)}")
        
    return result

if __name__ == "__main__":
    import json
    sample_image = "test_news.png"
    
    if os.path.exists(sample_image):
        output = run_ocr_pipeline(sample_image)
        print("\n--- ENHANCED OCR OUTPUT ---")
        print(json.dumps(output, indent=4, ensure_ascii=False))
    else:
        print(f"Error: Could not find '{sample_image}' in the folder.")

        print(f"Error: Could not find '{sample_image}' in the folder.")