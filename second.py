import gc
import glob
import os
import ast
import joblib
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ====================== CONFIG ======================
TRAIN_PATH = "./DataSet/release_train_patients"
VALID_PATH = "./DataSet/release_validate_patients"
TEST_PATH = "./DataSet/release_test_patients"
METADATA_PATH = "lgbm_ddxplus_full_meta.joblib"

MAX_TRAIN_ROWS = None
MAX_VALID_ROWS = None
MAX_TEST_ROWS = None
TOP_EVIDENCES = None
# ===================================================


def parse_evidences(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []


def load_frame(path: str, usecols=None, max_rows=None):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv")) + glob.glob(
            os.path.join(path, "*.parquet")
        )
        frames = []
        remaining = max_rows
        for f in sorted(files):
            per_max = remaining if remaining else None
            if f.endswith(".parquet"):
                df = pd.read_parquet(f, columns=usecols)
            else:
                df = pd.read_csv(f, usecols=usecols, nrows=per_max)
            frames.append(df)
            if remaining:
                remaining -= len(df)
                if remaining <= 0:
                    break
        return pd.concat(frames, ignore_index=True)
    else:
        if path.endswith(".parquet"):
            return pd.read_parquet(path, columns=usecols)
        return pd.read_csv(path, usecols=usecols, nrows=max_rows)


def preprocess_data(df: pd.DataFrame, vocab=None):
    df = df.copy()

    # Parse evidences
    df["EVIDENCES"] = df["EVIDENCES"].apply(parse_evidences)

    # Build vocabulary (only on train)
    if vocab is None:
        counter = Counter()
        for lst in df["EVIDENCES"]:
            counter.update(lst)
        if TOP_EVIDENCES is None:
            vocab = list(counter.keys())
        else:
            vocab = [item for item, _ in counter.most_common(TOP_EVIDENCES)]
        print(f"✅ Vocabulary created: {len(vocab)} evidences")

    # Binary features (int8 = very low memory)
    for ev in vocab:
        df[f"ev_{ev}"] = (
            df["EVIDENCES"].apply(lambda x: 1 if ev in x else 0).astype("int8")
        )

    df.drop(columns=["EVIDENCES"], inplace=True)

    # === CRITICAL FIX FOR LIGHTGBM ===
    return df, vocab


def apply_categories(
    df: pd.DataFrame,
    sex_categories: list[str],
    init_categories: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df["SEX"] = df["SEX"].where(df["SEX"].isin(sex_categories), "__OTHER__")
    df["INITIAL_EVIDENCE"] = df["INITIAL_EVIDENCE"].where(
        df["INITIAL_EVIDENCE"].isin(init_categories), "__OTHER__"
    )
    df["SEX"] = pd.Categorical(df["SEX"], categories=sex_categories)
    df["INITIAL_EVIDENCE"] = pd.Categorical(
        df["INITIAL_EVIDENCE"], categories=init_categories
    )
    return df


def evaluate_model(name: str, y_true, y_pred):
    print(f"\n=== {name.upper()} METRICS ===")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.5f}")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_true, y_pred):.5f}")
    print(
        f"F1 Macro          : {f1_score(y_true, y_pred, average='macro', zero_division=0):.5f}"
    )
    print(
        f"F1 Weighted       : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.5f}"
    )


def main():
    print("=== DDXPlus LightGBM - Full Dataset (CPU) ===\n")

    # Load Train
    train_cols = ["AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE", "PATHOLOGY"]
    train_df = load_frame(TRAIN_PATH, usecols=train_cols, max_rows=MAX_TRAIN_ROWS)
    print(f"Loaded train: {len(train_df):,} rows")

    train_df, vocab = preprocess_data(train_df)
    sex_categories = sorted(train_df["SEX"].dropna().unique().tolist())
    init_categories = sorted(train_df["INITIAL_EVIDENCE"].dropna().unique().tolist())
    if "__OTHER__" not in sex_categories:
        sex_categories.append("__OTHER__")
    if "__OTHER__" not in init_categories:
        init_categories.append("__OTHER__")
    train_df = apply_categories(train_df, sex_categories, init_categories)
    X_train = train_df.drop(columns=["PATHOLOGY"])
    y_train = train_df["PATHOLOGY"]

    # Load Valid
    X_valid, y_valid = None, None
    if os.path.exists(VALID_PATH):
        valid_df = load_frame(VALID_PATH, usecols=train_cols, max_rows=MAX_VALID_ROWS)
        print(f"Loaded valid: {len(valid_df):,} rows")
        valid_df, _ = preprocess_data(valid_df, vocab=vocab)
        valid_df = apply_categories(valid_df, sex_categories, init_categories)
        X_valid = valid_df.drop(columns=["PATHOLOGY"])
        y_valid = valid_df["PATHOLOGY"]

    # LightGBM Model (full server, CPU only)
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        max_bin=255,
    )

    print("Training LightGBM...")
    eval_set = [(X_valid, y_valid)] if X_valid is not None else None
    callbacks = [lgb.early_stopping(40, verbose=True)] if eval_set else None

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="multi_logloss",
        callbacks=callbacks,
        categorical_feature=["SEX", "INITIAL_EVIDENCE"],
    )

    # Evaluate
    evaluate_model("Train", y_train, model.predict(X_train))
    if X_valid is not None:
        evaluate_model("Valid", y_valid, model.predict(X_valid))

    # Save
    joblib.dump(model, "lgbm_ddxplus_full.joblib")
    joblib.dump(
        {
            "vocab": vocab,
            "sex_categories": sex_categories,
            "init_categories": init_categories,
        },
        METADATA_PATH,
    )
    print("\n✅ Model saved as 'lgbm_ddxplus_full.joblib'")
    print(f"✅ Metadata saved as '{METADATA_PATH}'")

    gc.collect()
    print("Done! Used full dataset and CPU resources.")


if __name__ == "__main__":
    main()
