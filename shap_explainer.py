# shap_explainer.py
import shap
import numpy as np

def get_shap_explanations(model, sample_df, feature_names, top_k=4):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    # Take absolute values & pick top highest contributors
    shap_val = np.abs(shap_values).flatten()
    top_idx = np.argsort(shap_val)[-top_k:][::-1]

    explanations = []
    for i in top_idx:
        explanations.append(f"{feature_names[i]} influenced the model")

    return explanations
