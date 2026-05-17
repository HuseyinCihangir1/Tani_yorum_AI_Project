import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ethics import review_text


BASE_DIR = Path(__file__).resolve().parent
FINETUNED_MODEL_DIR = BASE_DIR / "models" / "house_bert_model_finetuned"
AUGMENTED_MODEL_DIR = BASE_DIR / "models" / "house_bert_model_improved_augmented"
IMPROVED_MODEL_DIR = BASE_DIR / "models" / "house_bert_model_improved"
FALLBACK_MODEL_DIR = BASE_DIR / "models" / "house_bert_model"
MODEL_CANDIDATES = (
    FINETUNED_MODEL_DIR,
    AUGMENTED_MODEL_DIR,
    IMPROVED_MODEL_DIR,
    FALLBACK_MODEL_DIR,
)
EXPECTED_DATA_PIPELINE_VERSION = "symptom_exploded_canonical_v3"

METADATA_PATH = BASE_DIR / "data" / "processed" / "preprocess_metadata.json"
LEGACY_MAPPING_PATH = BASE_DIR / "data" / "processed" / "bert_top100_symptom_label_mapping.json"
MAX_LENGTH = 128
OTHER_LABEL = "diğer"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def choose_model_dir() -> Path:
    required_files = ("config.json", "tokenizer.json")
    weight_files = ("model.safetensors", "pytorch_model.bin")
    eligible_models = []

    for candidate in MODEL_CANDIDATES:
        if not candidate.exists():
            continue

        has_required = all((candidate / file_name).exists() for file_name in required_files)
        has_weights = any((candidate / file_name).exists() for file_name in weight_files)
        if not has_required or not has_weights:
            continue

        metrics = read_json(candidate / "training_metrics.json")
        test_metrics = metrics.get("test_metrics", {})
        quality_score = float(test_metrics.get("weighted_f1", test_metrics.get("accuracy", 0.0)))
        pipeline_match = int(metrics.get("data_pipeline_version") == EXPECTED_DATA_PIPELINE_VERSION)
        eligible_models.append(
            (pipeline_match, quality_score, -MODEL_CANDIDATES.index(candidate), candidate)
        )

    if not eligible_models:
        return FALLBACK_MODEL_DIR

    return max(eligible_models)[3]


MODEL_DIR = choose_model_dir()
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"


st.set_page_config(
    page_title="House MD BERT Analiz",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 650;
    }
    .ethics-note {
        border-left: 4px solid #1f77b4;
        background: #f6f8fb;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_unicode(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return value.replace("i\u0307", "i")


def prepare_model_text(value: str) -> str:
    text = normalize_unicode(str(value))
    return re.sub(r"\s+", " ", text).strip()


def load_label_names(model_config, mapping_path: Path) -> list[str]:
    mapping = read_json(mapping_path) or read_json(LEGACY_MAPPING_PATH)
    id_to_label = mapping.get("id_to_label", {})

    if id_to_label:
        return [
            id_to_label.get(str(index), f"LABEL_{index}")
            for index in range(len(id_to_label))
        ]

    config_labels = getattr(model_config, "id2label", {})
    return [
        config_labels.get(index, config_labels.get(str(index), f"LABEL_{index}"))
        for index in range(getattr(model_config, "num_labels", len(config_labels)))
    ]


@st.cache_resource(show_spinner="BERT modeli yukleniyor...")
def load_bert_model(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model klasoru bulunamadi: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    labels = load_label_names(model.config, model_path / "label_mapping.json")
    return tokenizer, model, labels, device


def predict(text: str, top_k: int) -> tuple[pd.DataFrame, str]:
    tokenizer, model, labels, device = load_bert_model(str(MODEL_DIR))
    prepared_text = prepare_model_text(text)

    inputs = tokenizer(
        prepared_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        bert_probs = torch.softmax(logits, dim=-1)

    top_k = min(top_k, bert_probs.shape[-1])
    scores, indices = torch.topk(bert_probs, top_k)

    rows = []
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        label = labels[index] if index < len(labels) else f"LABEL_{index}"
        rows.append(
            {
                "Sira": rank,
                "Semptom": label,
                "BERT (%)": round(score * 100, 2),
                "Label ID": index,
            }
        )

    return pd.DataFrame(rows), prepared_text


metadata = read_json(METADATA_PATH)
training_metrics = read_json(TRAINING_METRICS_PATH)

st.title("House MD BERT Klinik Metin Analizi")
st.caption("Sadece BERT tabanli semptom siniflandirma prototipi")

st.markdown(
    """
    <div class="ethics-note">
    Bu uygulama egitim ve proje gosterimi icindir. Tani koymaz, tedavi onermez ve doktor degerlendirmesinin yerine gecmez.
    Acil veya hayati risk tasiyan durumlarda 112 ya da en yakin acil servis esas alinmalidir.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Etik Sinirlar")
    st.write("Tani ve tedavi karari uretmez.")
    st.write("Kullanici metnini dosyaya kaydetmez.")
    st.write("Kisisel veri ve acil risk uyarisi verir.")
    st.write("Model ciktisi yalnizca BERT siniflandirma sonucudur.")

    st.divider()
    st.header("Model")
    st.write(f"Model: `{MODEL_DIR.name}`")
    st.write("Girdi: `text`")
    st.write("Hedef: `Symptom`")
    st.write("Tahmin: Sadece BERT")
    st.write("Kapsam: top 30 semptom + `diger`")

    if training_metrics:
        split_rows = training_metrics.get("split_rows", {})
        st.metric("Model veri satiri", training_metrics.get("row_count", "-"))
        st.metric("Model sinifi", training_metrics.get("class_count", "-"))
        if training_metrics.get("uses_other_label"):
            st.metric("Diger sinif satiri", training_metrics.get("other_class_rows", "-"))
        if split_rows:
            st.write(
                "Split: "
                f"train `{split_rows.get('train', '-')}`, "
                f"val `{split_rows.get('validation', '-')}`, "
                f"test `{split_rows.get('test', '-')}`"
            )
        test_metrics = training_metrics.get("test_metrics", {})
        if test_metrics:
            st.metric("Test accuracy", f"{test_metrics.get('accuracy', 0):.2%}")
            st.metric("Test top-3", f"{test_metrics.get('top_3_accuracy', 0):.2%}")
    elif metadata:
        st.metric("Islenmis satir", metadata.get("processed_rows", "-"))
        feature_selection = metadata.get("feature_selection", {})
        st.metric("BERT top-N satir", feature_selection.get("bert_top_n_rows", "-"))
        st.metric("Sinif sayisi", feature_selection.get("bert_top_n_class_count", "-"))

left_col, right_col = st.columns([1.25, 0.75], gap="large")

with left_col:
    user_text = st.text_area(
        "Diyalog metni",
        height=210,
        placeholder="Orn: Hasta ates, oksuruk ve nefes darligi yasiyor.",
    )

    control_col_1, control_col_2 = st.columns([0.35, 0.65])
    with control_col_1:
        top_k = st.slider("Tahmin", min_value=3, max_value=10, value=5, step=1)
    with control_col_2:
        analyze_clicked = st.button("Analiz Et", type="primary")

with right_col:
    st.subheader("Veri Durumu")
    if metadata:
        dropped_rows = metadata.get("dropped_rows", {})
        class_distribution = metadata.get("class_distribution", {})
        st.metric("Ham satir", metadata.get("raw_rows", "-"))
        st.metric("Atilan satir", dropped_rows.get("total", "-"))
        st.metric("Toplam semptom sinifi", class_distribution.get("total_classes", "-"))
    else:
        st.warning("On isleme metadata dosyasi bulunamadi.")

if user_text.strip():
    ethics = review_text(user_text)

    if ethics.has_personal_data:
        st.warning(
            "Kisisel veri benzeri ifade algilandi: "
            + ", ".join(ethics.personal_data_matches)
            + ". Proje kullanimi icin bu bilgileri metinden kaldirin."
        )

    if ethics.has_emergency_risk:
        st.error(
            "Acil risk ifadesi algilandi: "
            + ", ".join(ethics.emergency_matches)
            + ". Bu durumda model ciktisina gore hareket edilmez; 112 veya acil servis yonlendirmesi esastir."
        )

if analyze_clicked:
    if not user_text.strip():
        st.error("Analiz icin metin girilmelidir.")
    else:
        try:
            predictions, prepared_text = predict(user_text, top_k)
        except Exception as error:
            st.error(f"Model calistirilamadi: {error}")
        else:
            st.subheader("BERT Tahminleri")
            st.dataframe(
                predictions,
                hide_index=True,
                use_container_width=True,
            )

            chart_data = predictions.set_index("Semptom")["BERT (%)"]
            st.bar_chart(chart_data)

            top_label = str(predictions.iloc[0]["Semptom"])
            top_score = float(predictions.iloc[0]["BERT (%)"])
            if top_label == OTHER_LABEL:
                st.info(
                    "En yuksek sonuc `diger`: metin top 30 semptom disinda veya egitimde seyrek kalan bir sinifa benziyor."
                )
            if top_score < 35:
                st.warning(
                    "BERT model guveni dusuk. Bu cikti yalnizca proje ici siniflandirma sinyali olarak degerlendirilmelidir."
                )

            with st.expander("BERT girdisi"):
                st.code(prepared_text, language="text")
