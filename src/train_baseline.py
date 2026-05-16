import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/ready_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/metrics")
RANDOM_STATE = 42


def normalize_label(value):
    return str(value).lower().strip()


def build_pipeline():
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def top_k_accuracy(y_true, scores, classes, k):
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])

    top_indices = np.argsort(scores, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean([truth in labels for truth, labels in zip(y_true, top_labels)]))


def select_training_rows(df, target_col, top_n, min_count):
    work = df[["clean_text", target_col]].dropna().copy()
    work["clean_text"] = work["clean_text"].astype(str).str.strip()
    work[target_col] = work[target_col].apply(normalize_label)
    work = work[(work["clean_text"] != "") & (work[target_col] != "")]

    counts = work[target_col].value_counts()
    selected_labels = counts[counts >= min_count]
    if top_n:
        selected_labels = selected_labels.head(top_n)

    work = work[work[target_col].isin(selected_labels.index)].copy()
    return work, selected_labels


def train_one_model(df, target_col, model_name, top_n, min_count):
    work, selected_labels = select_training_rows(df, target_col, top_n, min_count)
    if work[target_col].nunique() < 2:
        raise ValueError(f"{target_col} icin en az iki sinif gerekli.")

    x_train, x_test, y_train, y_test = train_test_split(
        work["clean_text"],
        work[target_col],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work[target_col],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    scores = model.decision_function(x_test)
    classes = model.classes_

    metrics = {
        "target": target_col,
        "model_name": model_name,
        "rows_used": int(len(work)),
        "class_count": int(work[target_col].nunique()),
        "top_n": top_n,
        "min_count": min_count,
        "test_size": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "top_3_accuracy": top_k_accuracy(y_test.to_numpy(), scores, classes, min(3, len(classes))),
    }

    text_report = classification_report(y_test, y_pred, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    metrics_path = REPORT_DIR / f"{model_name}_metrics.json"
    report_path = REPORT_DIR / f"{model_name}_classification_report.txt"
    labels_path = REPORT_DIR / f"{model_name}_label_distribution.csv"

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(text_report, encoding="utf-8")
    selected_labels.rename("count").to_csv(labels_path, encoding="utf-8")

    print(f"{model_name}: model kaydedildi -> {model_path}")
    print(
        f"{model_name}: accuracy={metrics['accuracy']:.3f}, "
        f"macro_f1={metrics['macro_f1']:.3f}, top_3={metrics['top_3_accuracy']:.3f}"
    )
    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Once src/preprocess.py calistirilmali.")

    df = pd.read_csv(DATA_PATH)
    all_metrics = [
        train_one_model(df, "Symptom", "symptom_classifier", top_n=50, min_count=5),
        train_one_model(df, "Emotion", "emotion_classifier", top_n=10, min_count=10),
    ]

    metadata_path = MODEL_DIR / "baseline_metadata.json"
    metadata_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model ozeti kaydedildi -> {metadata_path}")


if __name__ == "__main__":
    main()
