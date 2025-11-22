import pandas as pd
import numpy as np
import tldextract
import re
import math
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------

def shannon_entropy(s):
    if not s:
        return 0
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log(p, 2) for p in probabilities])

def has_ip(url):
    """Check if domain is an IP address"""
    domain = urlparse(url).netloc
    return bool(re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", domain))


# -----------------------------------------------------------
# 2. Main Feature Extraction Function
# -----------------------------------------------------------

def extract_features(url):
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)

        domain = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain
        path = parsed.path
        query = parsed.query

        # ------------------------------
        # 1. Lexical Features
        # ------------------------------
        url_len = len(url)
        num_dots = url.count('.')
        num_slashes = url.count('/')
        num_hyphens = url.count('-')
        num_digits = sum(c.isdigit() for c in url)
        special_chars = "@=?%&;_+*"
        num_special = sum(url.count(c) for c in special_chars)
        has_at = "@" in url
        uses_ip = has_ip(url)

        # ------------------------------
        # 2. Domain-based Features
        # ------------------------------
        domain_len = len(domain)
        subdomain_parts = subdomain.count('.') + 1 if subdomain else 0
        subdomain_len = len(subdomain)
        tld_type = suffix
        has_https = parsed.scheme == "https"
        has_port = ":" in parsed.netloc

        # ------------------------------
        # 3. Path & Query Features
        # ------------------------------
        path_len = len(path)
        query_len = len(query)
        query_params = query.count('&') + 1 if query else 0

        # keyword flags
        keywords = ["login", "verify", "account", "update", "secure", "free", "bonus", "promo"]
        keyword_flags = {f"kw_{kw}": int(kw in url.lower()) for kw in keywords}

        # ------------------------------
        # 4. Statistical Features
        # ------------------------------
        entropy = shannon_entropy(url)
        digit_ratio = num_digits / url_len
        special_ratio = num_special / url_len

        # ------------------------------
        # 5. Token Features
        # ------------------------------
        tokens = re.split(r'[./?=\-_%]', url)
        tokens = [t for t in tokens if t]
        num_tokens = len(tokens)
        avg_token_len = np.mean([len(t) for t in tokens]) if tokens else 0

        # ------------------------------
        # Combine All Features
        # ------------------------------
        features = {
            "url_length": url_len,
            "num_dots": num_dots,
            "num_slashes": num_slashes,
            "num_hyphens": num_hyphens,
            "num_digits": num_digits,
            "num_special_chars": num_special,
            "has_at_symbol": int(has_at),
            "uses_ip_domain": int(uses_ip),

            "domain_length": domain_len,
            "subdomain_count": subdomain_parts,
            "subdomain_length": subdomain_len,
            "tld": tld_type,
            "has_https": int(has_https),
            "has_port": int(has_port),

            "path_length": path_len,
            "query_length": query_len,
            "query_params": query_params,

            "entropy": entropy,
            "digit_ratio": digit_ratio,
            "special_ratio": special_ratio,

            "num_tokens": num_tokens,
            "avg_token_len": avg_token_len,
        }

        # add keyword features
        features.update(keyword_flags)

        return features

    except Exception as e:
        print("Error:", e, "URL:", url)
        return None
    
def encode_tld(df, mapping):
    global_mean = np.mean(list(mapping.values()))
    df["tld_enc"] = df["tld"].map(mapping).fillna(global_mean)
    df.drop(columns=["tld"], inplace=True)
    return df