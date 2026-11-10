# ==============================================================
# Unified Multilingual Fake News Detector — v2
# Integrated engine: URL Metadata + Image OCR + Text Intelligence
# + Live News Verification with Weighted Score Fusion
# ==============================================================

import streamlit as st
import joblib
import re
import os
import json
import shutil
import time
import pandas as pd
from gnews import GNews
from langdetect import detect, DetectorFactory

from metadata_analyzer import analyze_url_metadata
from ocr_pipeline import run_ocr_pipeline
from intelligence import check_domain_safety, analyze_text_sensationalism, is_trusted_publisher

DetectorFactory.seed = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="Multilingual Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ========================== CUSTOM CSS ==========================
st.markdown("""
<style>
    .verdict-card {
        padding: 30px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .verdict-fake {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #f39c12, #e67e22);
    }
    .verdict-real {
        background: linear-gradient(135deg, #27ae60, #229954);
    }
    .verdict-card h1 {
        margin: 0; font-size: 2.4rem; font-weight: 800;
    }
    .verdict-card p {
        font-size: 1.1rem; margin-top: 8px; opacity: 0.92;
    }
    .reason-item {
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #e74c3c;
        background: #fff5f5;
        border-radius: 0 8px 8px 0;
        color: #2c3e50 !important;
    }
    .reason-item strong, .reason-item span {
        color: #2c3e50 !important;
    }
    .reason-item-warn {
        border-left-color: #f39c12;
        background: #fef9e7;
        color: #2c3e50 !important;
    }
    .reason-item-warn strong, .reason-item-warn span {
        color: #2c3e50 !important;
    }
    .reason-item-info {
        border-left-color: #3498db;
        background: #ebf5fb;
        color: #2c3e50 !important;
    }
    .reason-item-info strong, .reason-item-info span {
        color: #2c3e50 !important;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #334155 !important;
        margin-left: 6px;
    }
    /* Fixed Metric card styling compatible with both light and dark mode */
    div[data-testid="stMetric"] {
        background-color: #f8fafc !important;
        padding: 14px 16px !important;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================== CONSTANTS ==========================

# Engine weights for weighted score fusion
ENGINE_WEIGHTS = {
    "metadata":  0.25,
    "ml_style":  0.35,
    "live_ver":  0.30,
    "clickbait": 0.10,
}

STOPWORDS = {
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'the', 'a', 'an', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'into', 'that', 'this', 'it', 'has', 'have', 'had',
    'will', 'would', 'can', 'could', 'breaking', 'news', 'major', 'big', 'today', 'now', 'update',
    'shocking', 'viral', 'report', 'reports', 'said', 'says', 'across', 'country', 'moved', 'up',
    'का', 'की', 'के', 'में', 'है', 'हैं', 'को', 'पर', 'से', 'और', 'एक', 'यह', 'कि', 'ने', 'था', 'थी', 'बड़ी', 'खबर',
    'आहे', 'आहेत', 'या', 'आणि', 'मध्ये', 'वर', 'सांगितले', 'केले', 'माहिती', 'बातमी', 'मोठी'
}

CRITICAL_PREDICATES = {
    'dead', 'dies', 'died', 'death', 'killed', 'assassinated', 'arrested', 'resigns', 'resigned',
    'banned', 'ban', 'collapsed', 'earthquake', 'cures', 'cured',
    'निधन', 'मृत्यु', 'मौत', 'गिरफ्तार', 'बंद', 'भूकंप', 'इलाज',
    'निधन', 'मृत्यू', 'अटक', 'बंदी', 'भूकंप'
}

REACTION_VETO_WORDS = {
    'grief', 'condoles', 'condolence', 'condolences', 'mourns', 'mourn', 'ex-gratia', 'ex gratia',
    'painful', 'saddened', 'tribute', 'homage', 'stampede', 'capsizes', 'tragedy', 'accident',
    'toll', 'injured', 'announces', 'relief', 'compensation', 'expresses', 'express', 'calls',
    'शोक', 'श्रद्धांजलि', 'मुआवजा', 'दुख', 'हादसा', 'घायल',
    'श्रद्धांजली', 'मदत', 'अपघात', 'दुःख', 'जखमी'
}

DEBUNK_MARKERS = {
    'fact check', 'fact-check', 'factcheck', 'pib fact check', 'fake claim',
    'hoax', 'debunked', 'debunk', 'false claim', 'morphed image', 'doctored', 'fake news',
    'फैक्ट चेक', 'फर्जी दावा', 'अफवाह', 'झूठा दावा', 'भ्रामक दावा',
    'फॅक्ट चेक', 'खोटा दावा', 'बनावट दावा'
}

NEGATION_MARKERS = {
    'denies', 'denied', 'deny', 'false', 'untrue', 'baseless', 'no evidence',
    'fabricated', 'misleading', 'incorrect', 'rubbishes', 'refutes', 'refuted',
    'नकारा', 'झूठ', 'अफवाह', 'खंडन', 'भ्रामक',
    'नाकारले', 'खोटे', 'अफवा'
}

# Phase 2 fix: targeted panic-hoax detection (no more generic words like "tax", "bank")
PANIC_URGENCY_WORDS = {
    'immediately', 'urgent', 'deadline', 'last date', 'warning', 'alert', 'hurry',
    'right now', 'act now', 'before midnight', 'hours left',
    'तुरंत', 'जल्दी', 'चेतावनी', 'अंतिम तारीख', 'सावधान', 'अभी',
    'तातडी', 'सावधान', 'शेवटची तारीख', 'लगेच'
}

PANIC_FINANCIAL_WORDS = {
    'blocked', 'deactivate', 'deactivated', 'suspend', 'suspended', 'freeze', 'frozen',
    'seized', 'confiscated', 'closed permanently', 'click this link', 'fill this form',
    'send otp', 'share otp', 'verify now',
    'ब्लॉक', 'निलंबित', 'बंद हो जाएगा', 'ओटीपी भेजें', 'लिंक पर क्लिक',
    'ब्लॉक', 'स्थगित', 'बंद होईल'
}

# ========================== MODEL LOADING ==========================

@st.cache_resource
def load_ml_model():
    """Load v2 calibrated pipeline if available, fallback to v1 (vectorizer + model)."""
    v2_path = os.path.join(BASE_DIR, "classifier_pipeline_v2.pkl")
    if os.path.exists(v2_path):
        pipeline = joblib.load(v2_path)
        return pipeline, "v2"
    # Fallback to v1
    vec = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "classifier_model.pkl"))
    return (vec, model), "v1"

def load_whitelist():
    try:
        with open(os.path.join(BASE_DIR, "domains_db.json"), "r") as f:
            db = json.load(f)
            return db.get("trusted", [])
    except Exception:
        return []

# ========================== HELPER FUNCTIONS ==========================

def detect_input_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang in ['hi', 'mr', 'en']:
            return lang
        return 'en'
    except Exception:
        return 'en'

def analyze_clickbait_and_normalize(text: str):
    """Detect clickbait formatting and return penalty + cleaned text."""
    reasons = []
    penalty = 0.0

    if re.search(r'([a-zA-Z])\1{2,}', text):
        penalty += 0.35
        reasons.append("Character elongation detected (e.g. 'PLEEEEASE')")

    alpha_words = [w for w in text.split() if w.isalpha() and len(w) > 2]
    if alpha_words:
        caps_ratio = sum(1 for w in alpha_words if w.isupper()) / len(alpha_words)
        if caps_ratio >= 0.35:
            penalty += 0.25
            reasons.append("Excessive ALL-CAPS sensationalism")

    if re.search(r'[!?]{2,}', text):
        penalty += 0.20
        reasons.append("Punctuation cluster detected (!! or ??)")

    normalized_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)
    return min(penalty, 0.85), reasons, normalized_text

def extract_core_tokens(text: str):
    """Extract content words, subject tokens, and predicates from text."""
    clean = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text.lower())
    tokens = [t for t in clean.split() if len(t) > 1]
    content_words = [t for t in tokens if t not in STOPWORDS]
    if not content_words:
        content_words = tokens
    predicates = set(tokens).intersection(CRITICAL_PREDICATES)
    subjects = [t for t in content_words if t not in predicates]
    return content_words, subjects, predicates

def predict_fake_probability(text: str, language: str, model_bundle, model_version: str) -> float:
    """Unified prediction interface — works with both v1 and v2 model formats."""
    if model_version == "v2":
        pipeline = model_bundle
        input_df = pd.DataFrame({'Text': [text], 'language': [language]})
        probs = pipeline.predict_proba(input_df)[0]
        return float(probs[1])  # P(fake)
    else:
        vec, model = model_bundle
        vec_input = vec.transform([text])
        probs = model.predict_proba(vec_input)[0]
        return float(probs[1])

def compute_fused_score(engine_scores: dict) -> float:
    """Weighted average across only the engines that actually fired."""
    active = {k: v for k, v in engine_scores.items() if v is not None}
    if not active:
        return 0.5
    total_weight = sum(ENGINE_WEIGHTS.get(k, 0.1) for k in active)
    if total_weight == 0:
        return 0.5
    fused = sum(ENGINE_WEIGHTS.get(k, 0.1) * active[k] for k in active) / total_weight
    return round(fused, 3)

def compute_verdict(fused_risk: float):
    """Three-tier verdict: REAL / SUSPICIOUS / FAKE."""
    if fused_risk >= 0.70:
        return "FAKE", "LIKELY FAKE NEWS"
    elif fused_risk >= 0.40:
        return "SUSPICIOUS", "UNVERIFIED / SUSPICIOUS"
    else:
        return "REAL", "LIKELY AUTHENTIC NEWS"

# ========================== LIVE VERIFICATION ==========================

def verify_claim_online(claim: str, lang: str = 'en', whitelist: list = None):
    """
    Improved live verification with:
    - Publisher trust weighting
    - Negation detection (headline contradicts claim)
    - Predicate coherence checking
    - Rate-limited GNews queries
    """
    safe_claim = re.sub(r'\s+', ' ', claim).strip().replace('"', '').replace("'", "")
    _, _, normalized_claim = analyze_clickbait_and_normalize(safe_claim)
    content_words, subjects, predicates = extract_core_tokens(normalized_claim)

    audit_log = []

    if not content_words:
        return "NO_COVERAGE", [], ["No meaningful keywords could be extracted from the claim."]

    # --- 1. Fact-check search (30-day window, broader) ---
    fc_client = GNews(language=lang, country='IN', period='30d', max_results=8)
    fc_query = " ".join(content_words[:5]) + " fact check"
    try:
        fc_articles = fc_client.get_news(fc_query)
    except Exception:
        fc_articles = []

    debunk_hits = []
    for art in fc_articles:
        raw_title = art.get('title', '')
        clean_title = raw_title.rsplit(' - ', 1)[0].lower()
        has_subject = any(s in clean_title for s in subjects[:4]) if subjects else True
        has_debunk = any(m in clean_title for m in DEBUNK_MARKERS)
        if has_subject and has_debunk:
            debunk_hits.append(art)
            audit_log.append(f"🚨 **Fact-Check Debunk Found:** *{raw_title}*")

    if debunk_hits:
        return "DEBUNKED", debunk_hits, audit_log

    # --- 2. Live news search (7-day window) ---
    news_client = GNews(language=lang, country='IN', period='7d', max_results=8)
    search_query = " ".join(content_words[:6]) if content_words else safe_claim[:80]
    try:
        time.sleep(0.8)  # Rate limiting between GNews calls
        live_articles = news_client.get_news(search_query)
    except Exception:
        live_articles = []

    if not live_articles:
        return "NO_COVERAGE", [], ["No mainstream media reports found in the past 7 days matching these keywords."]

    confirmed_articles = []
    trusted_confirmed = []
    query_terms = content_words[:6]

    for art in live_articles:
        raw_title = art.get('title', '')
        clean_title = raw_title.rsplit(' - ', 1)[0].lower()

        # Check for debunk markers in live results too
        if any(m in clean_title for m in DEBUNK_MARKERS) and any(s in clean_title for s in subjects[:4]):
            audit_log.append(f"🚨 **Fact-Check Article Found in Live News:** *{raw_title}*")
            return "DEBUNKED", [art], audit_log

        # Token overlap check
        matched_content = [w for w in query_terms if w in clean_title]
        overlap_ratio = len(matched_content) / max(len(query_terms), 1)
        if overlap_ratio < 0.40:
            audit_log.append(f"❌ **Rejected (Low Topic Overlap — {overlap_ratio*100:.0f}%):** *{raw_title}*")
            continue

        # Negation detection: headline contradicts the claim
        has_negation = any(neg in clean_title for neg in NEGATION_MARKERS)
        if has_negation and any(s in clean_title for s in subjects[:3]):
            audit_log.append(f"🔄 **Contradicting/Denying Report Found:** *{raw_title}*")
            return "DEBUNKED", [art], audit_log

        # Publisher trust check
        pub_title = art.get('publisher', {}).get('title', '')
        pub_href = art.get('publisher', {}).get('href', '')
        is_trusted = is_trusted_publisher(pub_title, pub_href, whitelist or [])

        confirmed_articles.append(art)
        if is_trusted:
            trusted_confirmed.append(art)

        trust_tag = " 🏛️" if is_trusted else ""
        audit_log.append(f"✅ **Corroborating Report{trust_tag}:** *{raw_title}*")

    if confirmed_articles:
        return "CONFIRMED_COVERAGE", confirmed_articles, audit_log
    else:
        return "UNRELATED_COVERAGE", live_articles, audit_log


# ========================== UI LAYOUT ==========================

# --- Header ---
st.title("📰 Multilingual Fake News Detector")
st.caption("Integrated engine combining **URL Metadata**, **Image OCR**, **Stylometric Text Intelligence**, and **Live News Verification** with weighted score fusion.")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ System Info")
    model_bundle, model_version = load_ml_model()
    version_label = "v2 (Calibrated Pipeline + Char N-grams)" if model_version == "v2" else "v1 (TF-IDF + LogReg)"
    st.markdown(f"**Model:** {version_label}")
    st.markdown("**Languages:** English · Hindi · Marathi")
    st.markdown("**Engines:** ML Classifier · Live News · URL Metadata · OCR")
    st.divider()
    st.markdown("##### How It Works")
    st.markdown("""
    1. **Text Intelligence** — ML classifier trained on 70k+ articles analyzes writing style
    2. **Live Verification** — Searches Google News for corroboration or debunking
    3. **URL Metadata** — Checks domain reputation, SSL, WHOIS age, author presence
    4. **Image OCR** — Extracts text from screenshots for analysis
    5. **Weighted Fusion** — Combines all active engine scores into a final verdict
    """)
    st.divider()
    st.caption("Scores from each engine are weighted and fused — not naively averaged.")

# --- Input Section ---
st.subheader("📥 Input")

text_to_analyze = st.text_area(
    "Enter a news headline, claim, or article text:",
    height=100,
    placeholder="Paste your news text here for analysis..."
)

with st.expander("📎 Additional inputs (optional — for deeper analysis)", expanded=False):
    col_url, col_img = st.columns(2)
    with col_url:
        url_to_analyze = st.text_input("🌐 News URL:", placeholder="https://example.com/article...")
    with col_img:
        uploaded_file = st.file_uploader("🖼️ News Screenshot:", type=["png", "jpg", "jpeg"])

# --- Analysis ---
if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):

    has_text = bool(text_to_analyze and text_to_analyze.strip())
    has_url = bool(url_to_analyze and url_to_analyze.strip())
    has_image = uploaded_file is not None

    if not has_text and not has_url and not has_image:
        st.warning("⚠️ Please provide at least one input — text, URL, or image.")
        st.stop()

    whitelist = load_whitelist()
    engine_scores = {}
    all_reasons = []      # list of (reason_text, source_label, severity)
    analysis_results = {}

    progress = st.progress(0, text="Initializing analysis pipeline...")

    try:
        # =================== ENGINE 1: URL METADATA ===================
        if has_url:
            progress.progress(10, text="🌐 Analyzing URL metadata & domain reputation...")

            meta_data = analyze_url_metadata(url_to_analyze)

            domain = meta_data.get("domain", "")
            domain_risk, domain_flags = check_domain_safety(domain, whitelist)

            # Use FULL extracted text, not snippet, for downstream analysis
            extracted_full_text = meta_data.get("extracted_data", {}).get("full_text", "") or ""
            text_risk, text_flags = analyze_text_sensationalism(extracted_full_text[:500])

            base_meta_score = meta_data.get("metadata_score", 0.0)
            final_meta_score = min(1.0, base_meta_score + domain_risk + text_risk)

            engine_scores["metadata"] = final_meta_score
            all_meta_flags = list(set(meta_data.get("flags", []) + domain_flags + text_flags))
            for f in all_meta_flags:
                all_reasons.append((f, "URL Metadata", "warn"))

            analysis_results["metadata"] = {
                "score": final_meta_score,
                "domain": domain,
                "author": meta_data.get("extracted_data", {}).get("author"),
                "date": meta_data.get("extracted_data", {}).get("date"),
                "title": meta_data.get("extracted_data", {}).get("title"),
                "snippet": meta_data.get("extracted_data", {}).get("text_snippet"),
                "flags": all_meta_flags,
                "expanded_url": meta_data.get("expanded_url", ""),
            }

            # Feed full extracted text to text pipeline if no direct text provided
            if not has_text and extracted_full_text:
                text_to_analyze = extracted_full_text
                has_text = True

        # =================== ENGINE 2: IMAGE OCR (extraction only) ===================
        if has_image:
            progress.progress(25, text="🖼️ Running OCR on uploaded image...")

            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file, buffer)

            ocr_data = run_ocr_pipeline(temp_file_path)

            ocr_text = ocr_data.get("extracted_text", "")
            ocr_conf = ocr_data.get("ocr_confidence", 0.0)
            ocr_flags = ocr_data.get("flags", [])

            for f in ocr_flags:
                all_reasons.append((f, "Image OCR", "info"))

            analysis_results["ocr"] = {
                "text": ocr_text,
                "confidence": ocr_conf,
                "flags": ocr_flags,
            }

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # Feed OCR text to text pipeline if no direct text provided
            if not has_text and ocr_text:
                text_to_analyze = ocr_text
                has_text = True

        # =================== ENGINE 3: TEXT INTELLIGENCE ===================
        if has_text and text_to_analyze.strip():
            progress.progress(40, text="📝 Analyzing text patterns & clickbait signals...")

            detected_lang = detect_input_language(text_to_analyze)

            # Clickbait analysis
            cb_penalty, cb_reasons, norm_text = analyze_clickbait_and_normalize(text_to_analyze)
            engine_scores["clickbait"] = cb_penalty
            for r in cb_reasons:
                all_reasons.append((r, "Clickbait Analysis", "warn"))

            # ML classifier
            progress.progress(55, text="🤖 Running ML classifier...")
            raw_ml_prob = predict_fake_probability(norm_text, detected_lang, model_bundle, model_version)
            ml_fake_prob = min(raw_ml_prob + cb_penalty, 0.99)
            ml_real_prob = 1.0 - ml_fake_prob
            engine_scores["ml_style"] = ml_fake_prob

            if ml_fake_prob >= 0.65:
                all_reasons.append((f"ML classifier rates this text as {ml_fake_prob*100:.0f}% likely fake", "ML Classifier", "error"))
            elif ml_fake_prob >= 0.40:
                all_reasons.append((f"ML classifier shows moderate suspicion ({ml_fake_prob*100:.0f}%)", "ML Classifier", "warn"))

            # Live verification
            progress.progress(65, text="📰 Verifying claim against live news sources...")
            live_status, articles, audit_log = verify_claim_online(
                text_to_analyze, lang=detected_lang, whitelist=whitelist
            )

            # Convert live status to engine score
            if live_status == "DEBUNKED":
                engine_scores["live_ver"] = 0.95
                all_reasons.append(("Fact-checking sources directly debunk this claim", "Live Verification", "error"))
            elif live_status == "CONFIRMED_COVERAGE":
                engine_scores["live_ver"] = 0.10
            elif live_status == "NO_COVERAGE":
                engine_scores["live_ver"] = 0.65
                all_reasons.append(("No mainstream news coverage found for this claim in the past 7 days", "Live Verification", "warn"))
            else:  # UNRELATED_COVERAGE
                engine_scores["live_ver"] = 0.50
                all_reasons.append(("News articles found but none directly corroborate this specific claim", "Live Verification", "warn"))

            # Improved panic-hoax check (Phase 2 fix)
            word_count = len(text_to_analyze.split())
            text_lower = text_to_analyze.lower()
            panic_financial = sum(1 for w in PANIC_FINANCIAL_WORDS if w in text_lower)
            panic_urgency = sum(1 for w in PANIC_URGENCY_WORDS if w in text_lower)
            if panic_financial >= 1 and panic_urgency >= 1 and word_count < 30 and live_status == "NO_COVERAGE":
                engine_scores["live_ver"] = 0.92
                all_reasons.append(("Financial panic / scam pattern detected: urgency + account threats + no news coverage", "Panic Hoax Detection", "error"))

            # Count trusted vs untrusted sources
            trusted_count = 0
            for art in articles:
                pub_title = art.get('publisher', {}).get('title', '')
                pub_href = art.get('publisher', {}).get('href', '')
                if is_trusted_publisher(pub_title, pub_href, whitelist):
                    trusted_count += 1

            analysis_results["text"] = {
                "language": detected_lang,
                "ml_fake_prob": ml_fake_prob,
                "ml_real_prob": ml_real_prob,
                "raw_ml_prob": raw_ml_prob,
                "clickbait_penalty": cb_penalty,
                "clickbait_reasons": cb_reasons,
                "live_status": live_status,
                "articles": articles,
                "trusted_count": trusted_count,
                "audit_log": audit_log,
                "word_count": word_count,
                "model_version": model_version,
            }

        # =================== SCORE FUSION & VERDICT ===================
        progress.progress(90, text="📊 Computing weighted verdict...")

        fused = compute_fused_score(engine_scores)
        verdict, verdict_label = compute_verdict(fused)

        progress.progress(100, text="✅ Analysis complete!")
        time.sleep(0.3)
        progress.empty()

        # =================== RESULTS DISPLAY ===================

        # --- Verdict Hero Card ---
        if verdict == "FAKE":
            card_class = "verdict-fake"
            card_emoji = "🚨"
        elif verdict == "SUSPICIOUS":
            card_class = "verdict-suspicious"
            card_emoji = "⚠️"
        else:
            card_class = "verdict-real"
            card_emoji = "✅"

        st.markdown(f"""
        <div class="verdict-card {card_class}">
            <h1>{card_emoji} {verdict_label}</h1>
            <p>Combined Risk Score: {fused*100:.1f}% — based on {len(engine_scores)} active engine{'s' if len(engine_scores) != 1 else ''}</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Why This Verdict ---
        if all_reasons:
            st.markdown("#### 🔍 Why This Verdict")
            for i, (reason, source, severity) in enumerate(all_reasons, 1):
                css_class = "reason-item" if severity == "error" else ("reason-item-warn" if severity == "warn" else "reason-item-info")
                st.markdown(f"""<div class="{css_class}">
                    <strong>{i}.</strong> {reason} <span class="source-badge">{source}</span>
                </div>""", unsafe_allow_html=True)

        # --- Source Cross-Reference ---
        if "text" in analysis_results:
            arts = analysis_results["text"].get("articles", [])
            trusted_ct = analysis_results["text"].get("trusted_count", 0)
            total_ct = len(arts)
            if total_ct > 0:
                st.info(f"📰 **Source Cross-Reference:** {total_ct} corroborating article{'s' if total_ct != 1 else ''} found, **{trusted_ct}** from trusted outlets.")
            else:
                st.warning("📰 **Source Cross-Reference:** No corroborating articles found from mainstream sources.")

        # --- Tabbed Detailed Results ---
        tab_labels = ["📊 Engine Scores"]
        if "text" in analysis_results:
            tab_labels.append("📝 Text Analysis")
        if "metadata" in analysis_results:
            tab_labels.append("🌐 URL Metadata")
        if "ocr" in analysis_results:
            tab_labels.append("🖼️ OCR Results")
        if "text" in analysis_results:
            tab_labels.append("🔍 Audit Trail")

        tabs = st.tabs(tab_labels)
        tab_idx = 0

        # Tab: Engine Scores
        with tabs[tab_idx]:
            score_cols = st.columns(len(engine_scores) + 1)
            label_map = {
                "metadata": "🌐 URL Metadata",
                "ml_style": "🤖 ML Classifier",
                "live_ver": "📰 Live Verification",
                "clickbait": "⚡ Clickbait Check",
            }
            for col, (key, score) in zip(score_cols, engine_scores.items()):
                weight = ENGINE_WEIGHTS.get(key, 0.1)
                with col:
                    st.metric(label_map.get(key, key), f"{score*100:.1f}%")
                    st.caption(f"Weight: {weight:.0%}")
            with score_cols[-1]:
                st.metric("🎯 Fused Score", f"{fused*100:.1f}%")
                st.caption("Weighted combination")
        tab_idx += 1

        # Tab: Text Analysis
        if "text" in analysis_results:
            with tabs[tab_idx]:
                t = analysis_results["text"]
                lang_map = {'en': 'English', 'hi': 'Hindi (हिंदी)', 'mr': 'Marathi (मराठी)'}
                st.caption(f"🌐 **Detected Language:** {lang_map.get(t['language'], 'English')} · **Model:** {t['model_version']} · **Word Count:** {t['word_count']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Stylometric Real Score", f"{t['ml_real_prob']*100:.1f}%")
                col2.metric("Stylometric Fake Risk", f"{t['ml_fake_prob']*100:.1f}%")
                col3.metric("Live Verification", t['live_status'].replace('_', ' '))

                if t['clickbait_reasons']:
                    st.warning("**Formatting Anomalies:** " + " · ".join(t['clickbait_reasons']))

                if t['clickbait_penalty'] > 0:
                    st.caption(f"ℹ️ Clickbait penalty of +{t['clickbait_penalty']*100:.0f}% was applied to the raw ML score ({t['raw_ml_prob']*100:.1f}%)")
            tab_idx += 1

        # Tab: URL Metadata
        if "metadata" in analysis_results:
            with tabs[tab_idx]:
                m = analysis_results["metadata"]
                st.metric("Metadata Risk Score", f"{m['score']*100:.1f}%")
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.markdown(f"**Domain:** `{m['domain']}`")
                meta_col2.markdown(f"**Author:** {m.get('author') or '❌ Not found'}")
                meta_col3.markdown(f"**Date:** {m.get('date') or '❌ Not found'}")

                if m.get('title'):
                    st.markdown(f"**Article Title:** {m['title']}")
                if m.get('snippet'):
                    with st.expander("Preview extracted text"):
                        st.write(m['snippet'])

                if m['flags']:
                    for flag in m['flags']:
                        st.warning(f"⚠️ {flag}")
            tab_idx += 1

        # Tab: OCR Results
        if "ocr" in analysis_results:
            with tabs[tab_idx]:
                o = analysis_results["ocr"]
                st.metric("OCR Confidence", f"{o['confidence']*100:.1f}%")
                st.caption("ℹ️ OCR is a text extraction step — the risk assessment is done by the text intelligence engine on the extracted text.")

                if o['text']:
                    st.text_area("Extracted Text:", o['text'], height=120, disabled=True)
                else:
                    st.info("No text could be extracted from the image.")

                if o['flags']:
                    for flag in o['flags']:
                        st.warning(f"⚠️ {flag}")
            tab_idx += 1

        # Tab: Audit Trail
        if "text" in analysis_results:
            with tabs[tab_idx]:
                t = analysis_results["text"]
                st.caption("Detailed log showing how each retrieved article was evaluated for relevance and credibility.")

                if t['audit_log']:
                    for entry in t['audit_log']:
                        st.markdown(f"- {entry}")
                else:
                    st.info("No audit entries were generated.")

                if t['articles']:
                    st.divider()
                    st.markdown("##### 📄 Retrieved Articles")
                    for idx, art in enumerate(t['articles'][:8], 1):
                        pub_title = art.get('publisher', {}).get('title', 'Unknown')
                        pub_href = art.get('publisher', {}).get('href', '')
                        is_trusted = is_trusted_publisher(pub_title, pub_href, whitelist)
                        trust_badge = " 🏛️" if is_trusted else ""

                        st.markdown(f"**{idx}. [{art.get('title')}]({art.get('url')})**{trust_badge}")
                        st.caption(f"Publisher: {pub_title}{' (Trusted)' if is_trusted else ''} · Published: {art.get('published date', 'Unknown')}")

    except Exception as e:
        progress.empty()
        st.error(f"⚠️ Analysis encountered an error: {str(e)}")
        st.info("Please try again. If the problem persists, try a different input or check your network connection (Live Verification requires internet access).")
