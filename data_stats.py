import json
from pathlib import Path

import pandas as pd

# =========================
# CONFIG
# =========================
DATA_PATH = "./DataSet/release_train_patients"  # change this
OUTPUT_JSON = "dataset_profile.json"
NROWS = 50000  # set to None to load full file
SAMPLE_VALUES_PER_COLUMN = 10


# =========================
# HELPERS
# =========================
def safe_convert(value):
    """Convert values to JSON-safe Python types."""
    if pd.isna(value):
        return None

    # numpy / pandas scalars
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return ext


def load_dataset(path: str, nrows=None) -> pd.DataFrame:
    ext = detect_file_type(path)

    if ext == "":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def summarize_numeric(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    return {
        "min": safe_convert(s.min()),
        "max": safe_convert(s.max()),
        "mean": safe_convert(s.mean()),
        "median": safe_convert(s.median()),
        "std": safe_convert(s.std()),
    }


def summarize_column(df: pd.DataFrame, col: str, sample_values_per_column: int):
    s = df[col]
    dtype_str = str(s.dtype)

    non_null = int(s.notna().sum())
    null_count = int(s.isna().sum())
    unique_count = int(s.nunique(dropna=True))

    # sample unique values
    sample_values = []
    try:
        uniques = s.dropna().astype(str).unique().tolist()[:sample_values_per_column]
        sample_values = uniques
    except Exception:
        sample_values = []

    top_values = {}
    try:
        vc = s.astype(str).value_counts(dropna=False).head(10)
        top_values = {str(k): int(v) for k, v in vc.items()}
    except Exception:
        top_values = {}

    numeric_summary = summarize_numeric(s)

    return {
        "name": col,
        "dtype": dtype_str,
        "non_null_count": non_null,
        "null_count": null_count,
        "null_ratio": null_count / len(df) if len(df) > 0 else None,
        "unique_count": unique_count,
        "unique_ratio": unique_count / non_null if non_null > 0 else None,
        "sample_values": sample_values,
        "top_values": top_values,
        "numeric_summary": numeric_summary,
    }


def detect_candidate_target_columns(df: pd.DataFrame):
    """
    Heuristic only:
    target-like columns are usually object/category columns
    with >1 and not-too-many unique values.
    """
    candidates = []

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        nunique = s.nunique(dropna=True)

        if nunique <= 1:
            continue

        is_object_like = (
            dtype_str == "object" or "string" in dtype_str or "category" in dtype_str
        )

        if is_object_like and nunique <= 200:
            candidates.append(
                {
                    "column": col,
                    "dtype": dtype_str,
                    "unique_count": int(nunique),
                    "top_values": {
                        str(k): int(v)
                        for k, v in s.astype(str)
                        .value_counts(dropna=False)
                        .head(10)
                        .items()
                    },
                }
            )

    return candidates


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("./DataSet/release_train_patients", nrows=NROWS)

    profile = {
        "file_path": DATA_PATH,
        "loaded_nrows": None if NROWS is None else NROWS,
        "actual_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows_count": int(df.duplicated().sum()),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "head_preview": [
            {k: safe_convert(v) for k, v in row.items()}
            for row in df.head(5).to_dict(orient="records")
        ],
        "columns_summary": [],
        "candidate_target_columns": detect_candidate_target_columns(df),
    }

    for col in df.columns:
        profile["columns_summary"].append(
            summarize_column(df, col, SAMPLE_VALUES_PER_COLUMN)
        )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"Done. JSON profile saved to: {OUTPUT_JSON}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Candidate target columns:")
    for item in profile["candidate_target_columns"][:10]:
        print("-", item["column"], "| unique_count =", item["unique_count"])


if __name__ == "__main__":
    main()
