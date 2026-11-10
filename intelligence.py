import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

# High-risk TLDs commonly used in malicious or clickbait domains
SUSPICIOUS_TLDS = {".xyz", ".top", ".loan", ".gq", ".cc", ".work", ".click", ".bid"}

# Multilingual Sensationalism & Clickbait Keywords (English, Hindi, Marathi)
SENSATIONAL_KEYWORDS = {
    "en": [
        "shocking", "miracle cure", "secret exposed", "they don't want you to know", 
        "mind blowing", "you won't believe", "exposed", "conspiracy", "banned"
    ],
    "hi": [
        "खौफनाक", "रहस्य", "खुलासा", "चमत्कारिक इलाज", "सच जानकर उड़ जाएंगे होश", 
        "वायरल सच", "बड़ा पर्दाफाश", "साजिश"
    ],
    "mr": [
        "धमकी", "रहस्य", "खुलासा", "चमत्कारिक उपाय", "सत्य जानकर थक्क व्हाल", 
        "व्हायरल सत्य", "मोता खुलासा", "कारस्थान"
    ]
}

def check_domain_safety(domain: str, whitelist: list) -> tuple:
    """
    Checks for suspicious TLDs and typosquatting against trusted whitelist domains.
    Returns (risk_modifier, flags_list)
    """
    risk_score = 0.0
    flags = []
    
    # 1. Check Suspicious TLD
    if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
        risk_score += 0.3
        flags.append(f"Suspicious high-risk TLD detected: {domain.split('.')[-1]}")
        
    # 2. Check Typo-Squatting / Lookalike domains against Whitelist
    for trusted in whitelist:
        if domain == trusted:
            return 0.0, [] # Exact match with trusted source
            
        # Calculate string similarity ratio
        similarity = SequenceMatcher(None, domain, trusted).ratio()
        if 0.8 <= similarity < 1.0:
            risk_score += 0.4
            flags.append(f"Potential typo-squatting domain imitating trusted outlet: {trusted}")
            break
            
    return risk_score, flags

def analyze_text_sensationalism(text: str) -> tuple:
    """
    Scans text for sensationalist keywords and punctuation abuse across English, Hindi, and Marathi.
    Returns (sensationalism_score, flags_list)
    """
    if not text:
        return 0.0, []
        
    score = 0.0
    flags = []
    text_lower = text.lower()
    
    # Check punctuation abuse (excessive exclamation marks or ALL CAPS words)
    exclamation_count = text.count("!")
    if exclamation_count > 3:
        score += 0.2
        flags.append("Excessive exclamation punctuation detected (clickbait pattern)")
        
    # Count keyword matches across all language dictionaries
    matched_keywords = []
    for lang, keywords in SENSATIONAL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched_keywords.append(kw)
                
    if matched_keywords:
        # Scale score based on number of trigger words found (capped at 0.5)
        keyword_penalty = min(0.5, len(matched_keywords) * 0.15)
        score += keyword_penalty
        flags.append(f"Sensationalist/emotional trigger words detected: {', '.join(set(matched_keywords[:4]))}")
        
    return min(1.0, score), flags


def is_trusted_publisher(publisher_name: str, publisher_href: str, whitelist: list) -> bool:
    """
    Check if a news article's publisher matches a trusted outlet from the whitelist.
    Works with GNews publisher metadata: {'title': 'NDTV', 'href': 'https://www.ndtv.com'}
    """
    if not whitelist:
        return False
    
    # Extract domain from publisher href URL
    if publisher_href:
        try:
            domain = urlparse(publisher_href).netloc.lower().replace("www.", "")
        except Exception:
            domain = ""
        if domain:
            for trusted in whitelist:
                if trusted in domain or domain in trusted:
                    return True
    
    # Fuzzy match: check publisher display name against whitelist domain bases
    if publisher_name:
        pub_lower = publisher_name.lower().strip()
        for trusted in whitelist:
            domain_base = trusted.split('.')[0].lower()
            if len(domain_base) > 3 and domain_base in pub_lower:
                return True
    
    return False
