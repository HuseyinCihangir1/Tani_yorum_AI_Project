import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/house_md_data.csv")
PROCESSED_DIR = Path("data/processed")
READY_DATA_PATH = PROCESSED_DIR / "ready_data.csv"
METADATA_PATH = PROCESSED_DIR / "preprocess_metadata.json"
FEATURE_REPORT_PATH = PROCESSED_DIR / "feature_selection_report.csv"
LABEL_MAPPING_PATH = PROCESSED_DIR / "symptom_label_mapping.json"
BERT_LABEL_MAPPING_PATH = PROCESSED_DIR / "bert_top100_symptom_label_mapping.json"

TARGET_COLUMN = "Symptom"
TEXT_COLUMN = "text"
BERT_TOP_N_LABELS = 100
TARGET_SPLIT_PATTERN = r"[,;]+"
DATA_PIPELINE_VERSION = "symptom_exploded_canonical_v3"

CANONICAL_LABELS = {
    "ateşli": "ateş",
    "yüksek ateş": "ateş",
    "enfarktüs": "kalp krizi",
    "enfarktus": "kalp krizi",
    "miyokard enfarktüsü": "kalp krizi",
    "miyokard enfarktusu": "kalp krizi",
    "heart attack": "kalp krizi",
    "kardiyak arrest": "kalp durması",
    "cardiac arrest": "kalp durması",
    "solunum sıkıntısı": "nefes darlığı",
    "solunum zorluğu": "nefes darlığı",
    "solunum rahatsızlığı": "nefes darlığı",
    "nefes alamama": "nefes darlığı",
    "nefes alma güçlüğü": "nefes darlığı",
    "öksürme": "öksürük",
    "öksürüyor": "öksürük",
    "seizure": "nöbet",
    "epileptik nöbet": "nöbet",
    "burun kanaması": "kanama",
    "kan kaybı": "kanama",
    "kanaması": "kanama",
    "pıhtılaşma": "pıhtı",
    "kan pıhtısı": "pıhtı",
    "trombus": "pıhtı",
    "trombüs": "pıhtı",
    "halusinasyon": "halüsinasyon",
    "inme": "felç",
    "paralizi": "felç",
    "böbrek fonksiyon kaybı": "böbrek yetmezliği",
    "karaciğer fonksiyon kaybı": "karaciğer yetmezliği",
}

EXPECTED_COLUMNS = [
    "season",
    "episode",
    "speaker",
    "Symptom",
    "Test",
    "Drug",
    "Procedure",
    "Intent",
    "diagnosis_stage",
    "Sarcasm",
    "Emotion",
    "Organ",
    "correct_prediction",
    "model_prediction",
    "text",
    "medical_entities",
]

NORMALIZED_LABEL_COLUMNS = [
    "Symptom",
    "Emotion",
    "Intent",
    "diagnosis_stage",
    "Organ",
    "correct_prediction",
    "model_prediction",
]

MISSING_LABEL_VALUES = {
    "",
    "-",
    "none",
    "nan",
    "null",
    "symptom",
    "belirti yok",
    "yok",
    "tanı",
    "tedavi",
    "alay",
}


def _normalize_unicode(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return value.replace("i\u0307", "i")


def clean_text(value) -> str:
    """Prepare the dialogue text for BERT tokenization."""
    if pd.isna(value):
        return ""

    text = _normalize_unicode(str(value)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(value):
    if pd.isna(value):
        return pd.NA

    label = _normalize_unicode(str(value)).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = CANONICAL_LABELS.get(label, label)
    if label in MISSING_LABEL_VALUES:
        return pd.NA

    return label


def split_target_labels(value) -> list[str]:
    if pd.isna(value):
        return []

    labels = []
    seen = set()
    for part in re.split(TARGET_SPLIT_PATTERN, str(value)):
        label = normalize_label(part)
        if pd.isna(label) or label in seen:
            continue
        seen.add(label)
        labels.append(label)

    return labels


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        _normalize_unicode(str(column)).replace("\ufeff", "").strip()
        for column in df.columns
    ]
    return df


def read_raw_data(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"'{path}' dosyasi bulunamadi.")

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "ISO-8859-9"):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="warn",
            )
            return normalize_columns(df), encoding
        except Exception as error:  # pragma: no cover - fallback diagnostics
            last_error = error

    raise RuntimeError(f"CSV okunamadi. Son hata: {last_error}")


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"Eksik kolonlar: {joined_columns}")


def build_label_mapping(labels: pd.Series) -> dict:
    categories = sorted(labels.dropna().unique().tolist())
    label_to_id = {label: index for index, label in enumerate(categories)}
    id_to_label = {str(index): label for label, index in label_to_id.items()}
    return {"label_to_id": label_to_id, "id_to_label": id_to_label}


def build_bert_label_mapping(df: pd.DataFrame, top_labels: pd.Index) -> dict:
    bert_df = df[df[TARGET_COLUMN].isin(top_labels)].copy()
    categories = pd.Categorical(bert_df[TARGET_COLUMN]).categories.tolist()
    label_to_id = {label: index for index, label in enumerate(categories)}
    id_to_label = {str(index): label for label, index in label_to_id.items()}

    return {
        "target": TARGET_COLUMN,
        "top_n": BERT_TOP_N_LABELS,
        "row_count": int(len(bert_df)),
        "class_count": int(len(categories)),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def preprocess_data() -> None:
    raw_df, used_encoding = read_raw_data(RAW_DATA_PATH)
    validate_schema(raw_df)

    prepared_df = raw_df.copy()
    raw_row_count = len(prepared_df)

    for column in NORMALIZED_LABEL_COLUMNS:
        if column in prepared_df.columns:
            prepared_df[column] = prepared_df[column].apply(normalize_label)

    prepared_df["clean_text"] = prepared_df[TEXT_COLUMN].apply(clean_text)

    target_labels = prepared_df[TARGET_COLUMN].apply(split_target_labels)
    has_text = prepared_df["clean_text"].str.len() > 0
    has_target = target_labels.str.len() > 0
    processed_df = prepared_df[has_text & has_target].copy()
    processed_df["_target_labels"] = target_labels[has_text & has_target]
    rows_before_explode = len(processed_df)
    processed_df = processed_df.explode("_target_labels").copy()
    processed_df[TARGET_COLUMN] = processed_df["_target_labels"]
    processed_df = processed_df.drop(columns=["_target_labels"]).reset_index(drop=True)

    label_mapping = build_label_mapping(processed_df[TARGET_COLUMN])
    processed_df["label"] = processed_df[TARGET_COLUMN].map(
        label_mapping["label_to_id"]
    )

    class_counts = processed_df[TARGET_COLUMN].value_counts()
    top_labels = class_counts.nlargest(BERT_TOP_N_LABELS).index
    feature_report = class_counts.rename_axis(TARGET_COLUMN).reset_index(name="count")
    feature_report["selected_for_bert_top100"] = feature_report[TARGET_COLUMN].isin(
        top_labels
    )
    feature_report["label"] = feature_report[TARGET_COLUMN].map(
        label_mapping["label_to_id"]
    )

    bert_label_mapping = build_bert_label_mapping(processed_df, top_labels)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "raw_data_path": str(RAW_DATA_PATH),
        "ready_data_path": str(READY_DATA_PATH),
        "encoding": used_encoding,
        "target_column": TARGET_COLUMN,
        "text_column": TEXT_COLUMN,
        "raw_rows": int(raw_row_count),
        "processed_rows": int(len(processed_df)),
        "rows_before_symptom_explode": int(rows_before_explode),
        "symptom_rows_added_by_explode": int(len(processed_df) - rows_before_explode),
        "dropped_rows": {
            "empty_text": int((~has_text).sum()),
            "missing_or_placeholder_target": int((has_text & ~has_target).sum()),
            "total": int(raw_row_count - len(processed_df)),
        },
        "columns": EXPECTED_COLUMNS + ["clean_text", "label"],
        "feature_selection": {
            "bert_input_feature": "clean_text",
            "target": TARGET_COLUMN,
            "retained_context_columns": [
                column
                for column in EXPECTED_COLUMNS
                if column not in {TEXT_COLUMN, TARGET_COLUMN}
            ],
            "excluded_from_model_input": [
                "correct_prediction",
                "model_prediction",
                "medical_entities",
            ],
            "bert_top_n_labels": BERT_TOP_N_LABELS,
            "bert_top_n_rows": bert_label_mapping["row_count"],
            "bert_top_n_class_count": bert_label_mapping["class_count"],
            "min_count_in_bert_top_n": int(
                class_counts.loc[top_labels].min() if len(top_labels) else 0
            ),
        },
        "class_distribution": {
            "total_classes": int(class_counts.size),
            "single_sample_classes": int((class_counts == 1).sum()),
            "classes_under_5_samples": int((class_counts < 5).sum()),
            "top_10": {
                label: int(count)
                for label, count in class_counts.head(10).to_dict().items()
            },
        },
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(READY_DATA_PATH, index=False, encoding="utf-8")
    feature_report.to_csv(FEATURE_REPORT_PATH, index=False, encoding="utf-8")
    write_json(LABEL_MAPPING_PATH, label_mapping)
    write_json(BERT_LABEL_MAPPING_PATH, bert_label_mapping)
    write_json(METADATA_PATH, metadata)

    print("BASARILI: 1. ve 2. asama veri hatti tamamlandi.")
    print(f"Ham satir sayisi: {raw_row_count}")
    print(f"Islenmis satir sayisi: {len(processed_df)}")
    print(f"Sinif sayisi: {class_counts.size}")
    print(f"BERT top-{BERT_TOP_N_LABELS} satir sayisi: {bert_label_mapping['row_count']}")
    print(f"Ciktilar: {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_data()
