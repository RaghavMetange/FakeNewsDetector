PBL26: Leakage-Aware Multilingual & Multi-Modal Fake News Detector

This project presents a free-to-run fake news detection framework for Hindi, Marathi, and English. The system analyzes direct text, news URLs, and screenshots in a single Streamlit application, fusing a calibrated text classifier, URL metadata, clickbait analysis, and live news verification into a comprehensive three-tier verdict (REAL, SUSPICIOUS, FAKE).

🛠️ Tech Stack & Architecture

The system relies on a weighted Bayesian fusion architecture rather than naïve score averaging, integrating the following tools:

Frontend UI: Streamlit for a unified, multi-modal web interface.

Backend API: FastAPI for serving REST endpoints (/analyze) that return JSON verdicts.

Machine Learning & NLP: scikit-learn drives the core logic, utilizing a Logistic Regression classifier fitted with a liblinear solver and balanced class weights, wrapped in CalibratedClassifierCV (sigmoid calibration). Feature extraction leverages sublinear TF-IDF across word (35,000 features) and character n-grams (20,000 features). Language detection is handled by langdetect.

Optical Character Recognition (OCR): EasyOCR extracts text from English, Hindi, and Marathi screenshots. Images are pre-processed using OpenCV with Contrast Limited Adaptive Histogram Equalization (CLAHE) and bilateral filtering to improve detection.

Web Scraping & URL Metadata: trafilatura extracts article bodies, while python-whois, BeautifulSoup, and requests evaluate SSL certificates, domain age, top-level domains, and look-alike domains.

Live News Verification: The GNews package queries Google News for the detected language. It performs a 30-day fact-check search and a 7-day corroborating coverage search to evaluate subject-predicate coherence and negate known hoaxes.

📦 Datasets & Models

To keep this repository lightweight and respect GitHub's file size limits, the large .csv datasets and trained .pkl models are not tracked in version control. You must download the data directly from the original public research repositories used to train this system:

Patil et al. Multilingual Dataset: Provides the core article-level fake and real corpora for Hindi and Marathi. Available on Zenodo (DOI: 10.5281/zenodo.11408513).

CONSTRAINT-2021: Supplies Hindi social media posts (retaining posts labeled 'fake' and 'non-hostile').

IFND (Indian Fake News Dataset): Provides over 56,000 Indian-context English headlines.

L3Cube-MahaNews: Provides Marathi real news records. (Note: The pipeline automatically cleans this dataset to remove source leakage cues like datelines and full-stop spacing habits prior to training).

Download the datasets, place them in the root directory, and run retrain_model.py to generate your local classifier_model.pkl and tfidf_vectorizer.pkl files.

🚀 Installation & Setup

1. Clone the Repository

git clone https://github.com/RaghavMetange/FakeNewsDetector.git
cd FakeNewsDetector


2. Install Dependencies
Ensure you have Python 3 installed, then install the required libraries:

pip install streamlit fastapi uvicorn scikit-learn joblib langdetect easyocr opencv-python-headless requests trafilatura beautifulsoup4 python-whois gnews pandas


3. Configure the Environment
Ensure your local folder contains the generated .pkl models from the training step, as well as domains_db.json (the trusted/flagged domain lists).

💻 Execution Instructions

The project features a decoupled frontend and backend. You must run them in separate terminal windows.

Step 1: Start the FastAPI Backend
Open your first terminal, ensure you are in the project root, and launch the API server:

uvicorn main_backend:app --reload --port 8000


The backend API will now be active at http://127.0.0.1:8000.

Step 2: Start the Streamlit Application
Open a second terminal window in the exact same project folder and launch the UI:

streamlit run app.py


Your browser will automatically open the application at http://localhost:8501.

🔎 Usage Guide

Select Input Mode: Use the dropdown in the Streamlit UI to select your input type: Direct Text, News URL, or Screenshot / Image.

Submit Claim: Paste the WhatsApp forward, news link, or upload the screenshot (supported formats: png, jpg, jpeg).

Review Verdict: The system will output a massive color-coded verdict card (REAL, SUSPICIOUS, or FAKE).

Audit the Evidence: Navigate through the dynamically generated tabs to view the text stylometric analysis, the URL metadata risk factors, OCR confidences, and the Live Verification audit trail showing exactly which Google News articles were referenced to corroborate or debunk the claim.
