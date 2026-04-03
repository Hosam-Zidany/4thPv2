import ast
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
)

# =========================
# 1) File paths
# =========================
TRAIN_PATH = "./DataSet/release_train_patients"
TEST_PATH = "./DataSet/release_test_patients"
VALID_PATH = "./DataSet/release_validate_patients"


# =========================
# 2) Load only simplified columns
#    We ignore DIFFERENTIAL_DIAGNOSIS on purpose
# =========================
USE_COLS = ["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"]

train_df = pd.read_csv(TRAIN_PATH, usecols=USE_COLS, nrows = 200000)
test_df = pd.read_csv(TEST_PATH, usecols=USE_COLS)
valid_df = pd.read_csv(VALID_PATH, usecols=USE_COLS)


# =========================
# 3) Basic cleaning
# =========================
train_df = train_df.drop_duplicates().reset_index(drop=True)
test_df = test_df.drop_duplicates().reset_index(drop=True)
valid_df = valid_df.drop_duplicates().reset_index(drop=True)


# =========================
# 4) Convert EVIDENCES from string to real Python list
#    Example:
#    "['E_48', 'E_50']"  ->  ['E_48', 'E_50']
# =========================
train_df["EVIDENCES"] = train_df["EVIDENCES"].apply(ast.literal_eval)
test_df["EVIDENCES"] = test_df["EVIDENCES"].apply(ast.literal_eval)
valid_df["EVIDENCES"] = valid_df["EVIDENCES"].apply(ast.literal_eval)


# =========================
# 5) Encode SEX
# =========================
sex_map = {"M": 0, "F": 1}

train_df["SEX"] = train_df["SEX"].map(sex_map)
test_df["SEX"] = test_df["SEX"].map(sex_map)
valid_df["SEX"] = valid_df["SEX"].map(sex_map)


# =========================
# 6) Convert EVIDENCES lists into binary columns
#    Fit only on training data, then transform test/valid
# =========================
mlb = MultiLabelBinarizer()

X_train_ev = mlb.fit_transform(train_df["EVIDENCES"])
X_test_ev = mlb.transform(test_df["EVIDENCES"])
X_valid_ev = mlb.transform(valid_df["EVIDENCES"])

X_train_ev = pd.DataFrame(X_train_ev, columns=mlb.classes_, index=train_df.index)
X_test_ev = pd.DataFrame(X_test_ev, columns=mlb.classes_, index=test_df.index)
X_valid_ev = pd.DataFrame(X_valid_ev, columns=mlb.classes_, index=valid_df.index)


# =========================
# 7) One-hot encode INITIAL_EVIDENCE
#    Fit on train, align test/valid to same columns
# =========================
train_init = pd.get_dummies(train_df["INITIAL_EVIDENCE"], prefix="INIT")
test_init = pd.get_dummies(test_df["INITIAL_EVIDENCE"], prefix="INIT")
valid_init = pd.get_dummies(valid_df["INITIAL_EVIDENCE"], prefix="INIT")

train_init, test_init = train_init.align(test_init, join="left", axis=1, fill_value=0)
train_init, valid_init = train_init.align(valid_init, join="left", axis=1, fill_value=0)

# After the second align, test may need realignment again to exactly match train
test_init = test_init.reindex(columns=train_init.columns, fill_value=0)
valid_init = valid_init.reindex(columns=train_init.columns, fill_value=0)


# =========================
# 8) Encode target PATHOLOGY
#    Fit on train only
# =========================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["PATHOLOGY"])
y_test = label_encoder.transform(test_df["PATHOLOGY"])
y_valid = label_encoder.transform(valid_df["PATHOLOGY"])


# =========================
# 9) Build final feature matrices
#    Simplified data:
#    AGE + SEX + INITIAL_EVIDENCE(one-hot) + EVIDENCES(binary)
# =========================
X_train_base = train_df[["AGE", "SEX"]].copy()
X_test_base = test_df[["AGE", "SEX"]].copy()
X_valid_base = valid_df[["AGE", "SEX"]].copy()

X_train = pd.concat([X_train_base, train_init, X_train_ev], axis=1)
X_test = pd.concat([X_test_base, test_init, X_test_ev], axis=1)
X_valid = pd.concat([X_valid_base, valid_init, X_valid_ev], axis=1)

# Safety: exact same column order
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Valid shape:", X_valid.shape)
print("Number of classes:", len(label_encoder.classes_))


# =========================
# 10) Train model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)


# =========================
# 11) Evaluation helper
# =========================
def evaluate_model(name, model, X, y_true, label_encoder):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    top3_acc = top_k_accuracy_score(
        y_true, y_proba, k=3, labels=np.arange(len(label_encoder.classes_))
    )

    print(f"\n========== {name} ==========")
    print(f"Accuracy           : {acc:.6f}")
    print(f"Balanced Accuracy  : {bal_acc:.6f}")
    print(f"Precision (macro)  : {precision_macro:.6f}")
    print(f"Recall (macro)     : {recall_macro:.6f}")
    print(f"F1-score (macro)   : {f1_macro:.6f}")
    print(f"Precision (weighted): {precision_weighted:.6f}")
    print(f"Recall (weighted)   : {recall_weighted:.6f}")
    print(f"F1-score (weighted) : {f1_weighted:.6f}")
    print(f"Top-3 Accuracy     : {top3_acc:.6f}")

    print(f"\nClassification Report ({name}):")
    print(
        classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix shape ({name}):", cm.shape)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "top3_accuracy": top3_acc,
    }


# =========================
# 12) Evaluate on train, test, validate
# =========================
train_metrics = evaluate_model("TRAIN", model, X_train, y_train, label_encoder)
test_metrics = evaluate_model("TEST", model, X_test, y_test, label_encoder)
valid_metrics = evaluate_model("VALIDATION", model, X_valid, y_valid, label_encoder)


# =========================
# 13) Compact final summary
# =========================
summary = pd.DataFrame(
    [train_metrics, test_metrics, valid_metrics], index=["TRAIN", "TEST", "VALIDATION"]
)

print("\n========== FINAL METRICS SUMMARY ==========")
print(summary)
