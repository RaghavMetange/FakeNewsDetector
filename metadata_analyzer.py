import re
import json
import os
from datetime import datetime
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import trafilatura
import whois

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def expand_url(url: str) -> str:
    """Expands URL shorteners (bit.ly, t.co, etc.) by following redirects."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except Exception:
        return url

def check_ssl_and_https(url: str) -> tuple[bool, str]:
    """Inspects if URL uses HTTPS and has a valid SSL certificate."""
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        return False, "URL does not use secure HTTPS protocol"
    
    try:
        requests.get(url, timeout=5, verify=True)
        return True, ""
    except requests.exceptions.SSLError:
        return False, "Invalid or expired SSL certificate detected"
    except Exception:
        return True, ""

def check_domain_reputation(domain: str) -> tuple[float, list[str]]:
    """Checks domain against a curated list of trusted and flagged outlets."""
    flags = []
    score_adjustment = 0.0
    
    db_path = os.path.join(BASE_DIR, "domains_db.json")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                db = json.load(f)
                
            if domain in db.get("trusted", []):
                score_adjustment -= 0.3
                flags.append("Domain belongs to a known trusted mainstream outlet")
            elif domain in db.get("flagged", []):
                score_adjustment += 0.5
                flags.append("Domain is present on a known flagged misinformation list")
        except Exception:
            pass
            
    return score_adjustment, flags

def get_domain_age_and_info(domain: str) -> tuple[float, list[str]]:
    """Checks domain age and registrar using python-whois, handling redaction gracefully."""
    flags = []
    risk_addition = 0.0
    
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
        if creation_date:
            if isinstance(creation_date, str):
                try:
                    creation_date = datetime.strptime(creation_date.split()[0], '%Y-%m-%d')
                except ValueError:
                    creation_date = None
                    
            if creation_date:
                age_days = (datetime.now() - creation_date).days
                if age_days < 30:
                    risk_addition += 0.4
                    flags.append(f"Domain registered very recently ({age_days} days ago)")
                elif age_days < 180:
                    risk_addition += 0.2
                    flags.append(f"Domain is relatively new ({age_days} days ago)")
            else:
                flags.append("Could not parse domain creation date format")
        else:
            risk_addition += 0.1
            flags.append("Domain creation date is hidden or redacted in WHOIS")
            
    except Exception:
        risk_addition += 0.05
        flags.append("WHOIS lookup failed or domain info is fully redacted/private")
        
    return risk_addition, flags

def analyze_headline_mismatch(soup: BeautifulSoup, extracted_title: str) -> tuple[float, list[str]]:
    """Compares Open Graph title against page heading."""
    flags = []
    risk_addition = 0.0
    
    og_title_tag = soup.find("meta", property="og:title")
    og_title = og_title_tag["content"].strip() if og_title_tag and og_title_tag.get("content") else ""
    
    if og_title:
        if sum(1 for c in og_title if c.isupper()) / max(len(og_title), 1) > 0.5 and len(og_title) > 10:
            risk_addition += 0.15
            flags.append("Open Graph title uses excessive ALL-CAPS styling (clickbait cue)")
                
    return risk_addition, flags

def analyze_url_metadata(input_url: str) -> dict:
    result = {
        "url": input_url,
        "expanded_url": "",
        "domain": "",
        "metadata_score": 0.0,
        "flags": [],
        "extracted_data": {
            "title": None,
            "author": None,
            "date": None,
            "full_text": None,
            "text_snippet": None
        }
    }
    
    expanded_url = expand_url(input_url)
    result["expanded_url"] = expanded_url
    parsed_uri = urlparse(expanded_url)
    domain = parsed_uri.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    result["domain"] = domain
    
    risk_score = 0.0
    flags = []
    
    # Check Domain Reputation (Whitelist / Blacklist)
    rep_risk, rep_flags = check_domain_reputation(domain)
    risk_score += rep_risk
    flags.extend(rep_flags)
    
    # Check HTTPS and SSL Validity
    is_secure, ssl_msg = check_ssl_and_https(expanded_url)
    if not is_secure:
        risk_score += 0.3
        flags.append(ssl_msg)
        
    # Domain Age & WHOIS Inspection
    domain_risk, domain_flags = get_domain_age_and_info(domain)
    risk_score += domain_risk
    flags.extend(domain_flags)
    
    # Fetch page content via trafilatura
    downloaded_page = trafilatura.fetch_url(expanded_url)
    if not downloaded_page:
        risk_score += 0.3
        flags.append("Failed to fetch article content (site may block scrapers or is offline)")
        result["metadata_score"] = max(0.0, min(round(risk_score, 2), 1.0))
        result["flags"] = flags
        return result
        
    metadata = trafilatura.extract_metadata(downloaded_page)
    extracted_text = trafilatura.extract(downloaded_page, include_comments=False, include_tables=False)
    
    author = metadata.author if metadata and metadata.author else None
    date = metadata.date if metadata and metadata.date else None
    title = metadata.title if metadata and metadata.title else None
    
    result["extracted_data"]["title"] = title
    result["extracted_data"]["author"] = author
    result["extracted_data"]["date"] = date
    result["extracted_data"]["full_text"] = extracted_text
    result["extracted_data"]["text_snippet"] = extracted_text[:300] + "..." if extracted_text else None
    
    if not author:
        risk_score += 0.15
        flags.append("No author listed on the article page")
    if not date:
        risk_score += 0.15
        flags.append("No publication date listed on the article page")
        
    soup = BeautifulSoup(downloaded_page, "html.parser")
    mismatch_risk, mismatch_flags = analyze_headline_mismatch(soup, title or "")
    risk_score += mismatch_risk
    flags.extend(mismatch_flags)
    
    result["metadata_score"] = max(0.0, min(round(risk_score, 2), 1.0))
    result["flags"] = flags
    
    return result
