# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_engineering import extract_features, encode_tld
from shap_explainer import get_shap_explanations
from db import init_db, save_prediction
import shap

# Initialize DB
init_db()

# ---------------------------
# Load Model + Preprocessing
# ---------------------------
lgb_model = lgb.Booster(model_file="models/lgbm_model.txt")

with open("models/tld_target_mapping.pkl", "rb") as f:
    tld_mapping = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

numeric_feature_list = [
    "url_length", "num_dots", "num_slashes", "num_hyphens",
    "num_digits", "num_special_chars", "has_at_symbol",
    "uses_ip_domain", "domain_length", "subdomain_count",
    "subdomain_length", "has_https", "has_port", "path_length",
    "query_length", "query_params", "entropy", "digit_ratio",
    "special_ratio", "num_tokens", "avg_token_len",
    "kw_login", "kw_verify", "kw_account", "kw_update",
    "kw_secure", "kw_free", "kw_bonus", "kw_promo",
    "tld_enc"
]

tfidf_feature_list = tfidf_vectorizer.get_feature_names_out().tolist()

# 3. Combine numeric + tfidf names
feature_columns = numeric_feature_list + tfidf_feature_list

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI()

class URLRequest(BaseModel):
    url: str


@app.post("/predict")
def predict_url(data: URLRequest):
    url = data.url

    # ---------------------------
    # Feature Engineering (numeric)
    # ---------------------------
    feats = extract_features(url)
    df_numeric = pd.DataFrame([feats])

    # ---------------------------
    # TLD Encoding
    # ---------------------------
    df_numeric = encode_tld(df_numeric, tld_mapping)

    # ---------------------------
    # TF-IDF Features
    # ---------------------------
    tfidf_vec = tfidf_vectorizer.transform([url]).toarray()

    # ---------------------------
    # Combine numeric + tfidf
    # ---------------------------
    df_numeric = df_numeric[numeric_feature_list]
    numeric_arr = df_numeric.values

    final_input = np.hstack([numeric_arr, tfidf_vec])


    # ---------------------------
    # Prediction
    # ---------------------------
    prob = float(lgb_model.predict(final_input)[0])
    pred = int(prob >= 0.5)

    # ---------------------------
    # Explanations
    # ---------------------------
    from explanation_engine import generate_explanations_from_shap
    sample_df=pd.DataFrame(final_input, columns=feature_columns)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(sample_df)

    reasons = generate_explanations_from_shap(
        feature_columns=feature_columns,
        shap_values=shap_values,
        numeric_features=numeric_feature_list,
        top_k=5
    )


    # ---------------------------
    # Save in Database
    # ---------------------------
    save_prediction(url, pred, prob)

    # ---------------------------
    # Response
    # ---------------------------
    return {
        "url": url,
        "prediction": pred,
        "probability": prob,
        "reasons": reasons
    }


# ---------------------------
# Run Locally
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
