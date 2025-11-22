# def generate_user_friendly_reasons(features, probability):

#     reasons = []

#     # 1. Suspicious keywords
#     suspicious_keywords = []
#     for kw in ["login", "verify", "account", "update", "secure", "free", "bonus", "promo"]:
#         if features.get(f"kw_{kw}", 0) == 1:
#             suspicious_keywords.append(kw)

#     if suspicious_keywords:
#         reasons.append(
#             f"Contains suspicious keywords: {', '.join(suspicious_keywords)}"
#         )

#     # 2. URL length
#     if features["url_length"] > 120:
#         reasons.append("The URL is unusually long, typical in phishing pages.")

#     # 3. Entropy (random characters)
#     if features["entropy"] > 4.0:
#         reasons.append("The URL contains random-looking characters (high entropy).")

#     # 4. TLD risk
#     risky_tlds = ["xyz", "top", "gq", "tk", "ml"]
#     if features["tld"] in risky_tlds:
#         reasons.append(f"The domain uses a high-risk TLD: .{features['tld']}")

#     # 5. No HTTPS
#     if features.get("has_https") == 0:
#         reasons.append("The website does not use HTTPS.")

#     # 6. Many subdomains
#     if features["subdomain_count"] > 2:
#         reasons.append(
#             "The URL has multiple subdomains, often used to mimic legitimate sites."
#         )

#     # 7. Many special characters
#     if features["num_special_chars"] > 20:
#         reasons.append("The URL contains too many special characters.")

#     # 8. Very high phishing probability
#     if probability > 0.98:
#         reasons.append("The URL strongly resembles known phishing patterns.")

#     # Fallback
#     if not reasons:
#         reasons.append("The URL exhibits patterns commonly seen in phishing attacks.")

#     return reasons

def categorize_feature(name, numeric_features):
    if name.startswith("kw_"):
        return "keyword"
    if name == "tld_enc":
        return "tld"
    if name in numeric_features:
        return "numeric"
    return "tfidf"


def natural_language_numeric(name):
    mapping = {
        "url_length": "The URL is unusually long",
        "num_dots": "The URL contains multiple dots, which is suspicious",
        "num_slashes": "The URL contains many slashes",
        "num_hyphens": "The URL has too many hyphens",
        "num_digits": "The URL contains many digits",
        "num_special_chars": "The URL contains many special characters",
        "entropy": "The URL has high randomness (entropy)",
        "subdomain_count": "The URL has too many subdomains",
        "domain_length": "The domain name is unusually long",
        "path_length": "The path section of the URL is very long",
    }
    return mapping.get(name, f"Anomalous numerical pattern: {name}")


def natural_language_keyword(name):
    word = name.replace("kw_", "")
    return f"The URL contains the suspicious keyword '{word}'"


def natural_language_tld():
    return "The top-level domain (TLD) is often used in phishing websites"


def natural_language_tfidf():
    return "The text structure of the URL resembles known phishing URLs"


def generate_explanations_from_shap(feature_columns, shap_values, numeric_features, top_k=5):

    shap_vals = abs(shap_values.flatten())
    top_indices = shap_vals.argsort()[-top_k:][::-1]

    reasons = set()

    for idx in top_indices:
        f = feature_columns[idx]
        cat = categorize_feature(f, numeric_features)

        if cat == "numeric":
            reasons.add(natural_language_numeric(f))

        elif cat == "keyword":
            reasons.add(natural_language_keyword(f))

        elif cat == "tld":
            reasons.add(natural_language_tld())

        elif cat == "tfidf":
            reasons.add(natural_language_tfidf())

    # If nothing detected
    if not reasons:
        reasons.add("The URL contains multiple phishing-like patterns")

    return list(reasons)
