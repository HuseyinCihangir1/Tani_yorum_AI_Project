import os
import re

import pandas as pd


INPUT_PATH = "data/raw/house_md_data.csv"
OUTPUT_PATH = "data/processed/ready_data.csv"
REQUIRED_COLUMNS = {"text", "Symptom"}
INVALID_LABELS = {"", "-", "none", "nan", "symptom", "belirsiz", "yok"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label):
    label = str(label).lower()
    label = re.sub(r"\s+", " ", label).strip()
    return label


def read_csv_safely(path):
    encodings = ("utf-8-sig", "utf-8", "ISO-8859-9")

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="warn",
            )
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        "CSV dosyasi utf-8-sig, utf-8 veya ISO-8859-9 olarak okunamadi.",
    )


def preprocess_data():
    if not os.path.exists(INPUT_PATH):
        print(f"HATA: '{INPUT_PATH}' dosyasi bulunamadi.")
        print("Beklenen konum: data/raw/house_md_data.csv")
        return False

    print(f"Dosya okunuyor: {INPUT_PATH}")
    df = read_csv_safely(INPUT_PATH)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        print(f"HATA: Eksik sutun(lar): {', '.join(sorted(missing_columns))}")
        print(f"Mevcut sutunlar: {', '.join(df.columns)}")
        return False

    before_count = len(df)
    df = df.dropna(subset=["text", "Symptom"]).copy()
    df["Symptom"] = df["Symptom"].apply(normalize_label)
    df = df[~df["Symptom"].isin(INVALID_LABELS)].copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""].copy()
    df["label"] = pd.Categorical(df["Symptom"]).codes

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"BASARILI: Veri islendi ve '{OUTPUT_PATH}' konumuna kaydedildi.")
    print(f"Satir sayisi: {before_count} -> {len(df)}")
    print(f"Sinif sayisi: {df['Symptom'].nunique()}")
    return True


if __name__ == "__main__":
    preprocess_data()
