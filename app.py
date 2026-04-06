import json
import joblib
import os

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# Load model and vocabulary
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "lgbm_ddxplus_full.joblib"))
metadata = joblib.load(os.path.join(BASE_DIR, "lgbm_ddxplus_full_meta.joblib"))
evidence_vocab = set(metadata["vocab"])
sex_categories = metadata["sex_categories"]
init_categories = metadata["init_categories"]

with open(os.path.join(BASE_DIR, "release_evidences.json"), "r") as f:
    EVIDENCE_MAP = json.load(f)

with open(os.path.join(BASE_DIR, "release_conditions.json"), "r") as f:
    CONDITIONS_MAP = json.load(f)

EVIDENCE_LABELS = {
    code: (data.get("question_en") or data.get("name") or code)
    for code, data in EVIDENCE_MAP.items()
}
CONDITION_LABELS = {
    key: (data.get("cond-name-eng") or data.get("condition_name") or key)
    for key, data in CONDITIONS_MAP.items()
}
print("Model loaded!")

# You'll need to save the evidence_vocab during training (we'll fix this)
# For now, we'll assume you re-create top evidences or load them


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/evidences", methods=["GET"])
def evidences():
    items = [{"code": code, "label": label} for code, label in EVIDENCE_LABELS.items()]
    items.sort(key=lambda x: x["label"].lower())
    return jsonify(items)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    age = int(data["age"])
    sex = data["sex"]
    initial_evidence = data["initial_evidence"]
    selected_evidences = data["evidences"]

    # Create feature row
    if sex not in sex_categories:
        sex = "__OTHER__"
    if initial_evidence not in init_categories:
        initial_evidence = "__OTHER__"

    row = {"AGE": age, "SEX": sex, "INITIAL_EVIDENCE": initial_evidence}

    # Add binary evidence features
    for ev in selected_evidences:
        if ev in evidence_vocab:
            row[f"ev_{ev}"] = 1

    df = pd.DataFrame([row])

    # Fill missing evidence columns with 0
    for col in model.feature_name_:
        if col not in df.columns and col.startswith("ev_"):
            df[col] = 0

    df["SEX"] = pd.Categorical(df["SEX"], categories=sex_categories)
    df["INITIAL_EVIDENCE"] = pd.Categorical(
        df["INITIAL_EVIDENCE"], categories=init_categories
    )

    df = df.reindex(columns=model.feature_name_, fill_value=0)

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    # Top 5 predictions
    top_indices = probabilities.argsort()[-5:][::-1]
    top_probs = probabilities[top_indices]

    return jsonify(
        {
            "predicted_pathology": CONDITION_LABELS.get(prediction, prediction),
            "top_5": [
                {
                    "pathology": CONDITION_LABELS.get(
                        model.classes_[i], model.classes_[i]
                    ),
                    "probability": float(p),
                }
                for i, p in zip(top_indices, top_probs)
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
