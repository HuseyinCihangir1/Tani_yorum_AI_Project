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
AUGMENTED_MODEL_DIR = BASE_DIR / "models" / "house_bert_model_improved_augmented"
IMPROVED_MODEL_DIR = BASE_DIR / "models" / "house_bert_model_improved"
FALLBACK_MODEL_DIR = BASE_DIR / "models" / "house_bert_model"
MODEL_CANDIDATES = (AUGMENTED_MODEL_DIR, IMPROVED_MODEL_DIR, FALLBACK_MODEL_DIR)


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

        metrics_path = candidate / "training_metrics.json"
        quality_score = -1.0
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                test_metrics = metrics.get("test_metrics", {})
                quality_score = float(
                    test_metrics.get("weighted_f1", test_metrics.get("accuracy", 0.0))
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                quality_score = -1.0

        eligible_models.append((quality_score, -MODEL_CANDIDATES.index(candidate), candidate))

    if not eligible_models:
        return FALLBACK_MODEL_DIR

    return max(eligible_models)[2]


MODEL_DIR = choose_model_dir()
MODEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"
LEGACY_MAPPING_PATH = BASE_DIR / "data" / "processed" / "bert_top100_symptom_label_mapping.json"
METADATA_PATH = BASE_DIR / "data" / "processed" / "preprocess_metadata.json"
TRAINING_METRICS_PATH = MODEL_DIR / "training_metrics.json"
PROTOTYPE_PATH = MODEL_DIR / "class_prototypes.pt"
MAX_LENGTH = 128

CLASSIFIER_WEIGHT = 0.30
PROTOTYPE_WEIGHT = 0.35
EXACT_MATCH_BASE_CONFIDENCE = 0.88
STRONG_MATCH_BASE_CONFIDENCE = 0.72
DISPLAY_EXCLUDED_LABELS = {"tanı"}


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


def fold_for_match(value: str) -> str:
    value = prepare_model_text(value).casefold()
    value = value.translate(
        str.maketrans(
            {
                "\u00e7": "c",
                "\u011f": "g",
                "\u0131": "i",
                "\u00f6": "o",
                "\u015f": "s",
                "\u00fc": "u",
            }
        )
    )
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_root(token: str) -> str:
    return token


def normalize_match_text(value: str) -> str:
    replacements = {
        "solumunda rahatsizlik": "nefes darligi",
        "solumu rahatsizlik": "nefes darligi",
        "solum rahatsizlik": "nefes darligi",
        "solum": "solunum",
        "solumu": "solunum",
        "solumunda": "solunumunda",
        "solunumda": "solunumunda",
        "solunumunda": "solunumunda",
        "solunumunda rahatsizlik": "nefes darligi",
        "solunum rahatsizlik": "nefes darligi",
        "solunum rahatsizligi": "nefes darligi",
        "nefes almakta zorlanma": "nefes darligi",
        "nefes alamama": "nefes darligi",
        "nefes sikintisi": "nefes darligi",
        "atesli": "ates",
        "atesim": "ates",
        "atesi": "ates",
        "oksuruyor": "oksuruk",
        "oksurme": "oksuruk",
        "kanamasi": "kanama",
        "kanamali": "kanama",
        "nobeti": "nobet",
        "nobetli": "nobet",
    }

    folded = fold_for_match(value)
    padded = f" {folded} "
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        padded = padded.replace(f" {source} ", f" {target} ")

    tokens = [token_root(token) for token in padded.split()]
    return " ".join(tokens)


LABEL_ALIASES = {
    "ates": ("ates", "atesli", "yuksek ates"),
    "nefes darligi": (
        "nefes darligi",
        "solunum rahatsizligi",
        "solunum sikintisi",
        "solunumunda rahatsizlik",
        "solumunda rahatsizlik",
        "nefes sikintisi",
    ),
    "oksuruk": ("oksuruk", "oksuruyor", "oksurme"),
    "kanama": ("kanama", "kanamasi", "siddetli kanama"),
    "nobet": ("nobet", "nobet gecirdi", "nobeti"),
    "tumor": ("tumor", "kitle"),
    "lezyon": ("lezyon",),
    "tasikardi": ("tasikardi", "kalp hizi", "hizli nabiz"),
    "kalp krizi": ("kalp krizi", "miyokard enfarktusu", "enfarktus"),
    "bas agrisi": ("bas agrisi",),
    "karin agrisi": ("karin agrisi",),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_prototypes(labels: list[str], device: torch.device, prototype_path: Path):
    if not prototype_path.exists():
        return None

    payload = torch.load(prototype_path, map_location=device)
    stored_labels = payload.get("labels", [])
    if stored_labels != labels:
        return None

    prototypes = payload["prototypes"].to(device)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    return prototypes


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
    prototypes = load_prototypes(labels, device, model_path / "class_prototypes.pt")
    return tokenizer, model, labels, device, prototypes


def lexical_scores(text: str, labels: list[str], device: torch.device) -> torch.Tensor:
    normalized_text = normalize_match_text(text)
    folded_text = f" {normalized_text} "
    text_tokens = set(normalized_text.split())
    scores = []

    for label in labels:
        folded_label = normalize_match_text(label)
        aliases = LABEL_ALIASES.get(folded_label, (folded_label,))
        alias_scores = []

        for alias in aliases:
            normalized_alias = normalize_match_text(alias)
            alias_tokens = [token for token in normalized_alias.split() if token]

            if not normalized_alias:
                alias_scores.append(0.0)
            elif f" {normalized_alias} " in folded_text:
                alias_scores.append(1.0)
            elif alias_tokens and all(token in text_tokens for token in alias_tokens):
                alias_scores.append(0.85)
            elif alias_tokens:
                overlap = sum(1 for token in alias_tokens if token in text_tokens)
                alias_scores.append(0.5 * (overlap / len(alias_tokens)))
            else:
                alias_scores.append(0.0)

        scores.append(max(alias_scores) if alias_scores else 0.0)

    return torch.tensor(scores, dtype=torch.float32, device=device)


def pooled_feature(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    encoder = getattr(model, model.base_model_prefix)
    outputs = encoder(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(pooled[0], dim=0)


def calibrated_confidence_scores(
    classifier_probs: torch.Tensor,
    prototype_probs: torch.Tensor,
    lex_scores: torch.Tensor,
    prototypes_available: bool,
) -> torch.Tensor:
    if prototypes_available:
        model_probs = (
            CLASSIFIER_WEIGHT * classifier_probs
            + PROTOTYPE_WEIGHT * prototype_probs
        )
        model_probs = model_probs / model_probs.sum()
        model_signal = torch.maximum(classifier_probs, prototype_probs)
    else:
        model_probs = classifier_probs
        model_signal = classifier_probs

    lex_confidence = torch.zeros_like(lex_scores)
    exact_mask = lex_scores >= 0.95
    strong_mask = (lex_scores >= 0.80) & ~exact_mask
    partial_mask = (lex_scores > 0) & ~exact_mask & ~strong_mask

    scaled_model_signal = torch.clamp(model_signal * 10.0, max=1.0)
    lex_confidence[exact_mask] = (
        EXACT_MATCH_BASE_CONFIDENCE + 0.08 * scaled_model_signal[exact_mask]
    )
    lex_confidence[strong_mask] = (
        STRONG_MATCH_BASE_CONFIDENCE + 0.12 * scaled_model_signal[strong_mask]
    )
    lex_confidence[partial_mask] = (
        0.05
        + 0.22 * lex_scores[partial_mask]
        + 0.05 * scaled_model_signal[partial_mask]
    )

    return torch.maximum(model_probs, lex_confidence).clamp(max=0.99)


def predict(text: str, top_k: int) -> tuple[pd.DataFrame, str]:
    tokenizer, model, labels, device, prototypes = load_bert_model(str(MODEL_DIR))
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
        classifier_probs = torch.softmax(logits, dim=-1)

        if prototypes is not None:
            feature = pooled_feature(model, inputs)
            prototype_scores = feature @ prototypes.T
            prototype_probs = torch.softmax(prototype_scores * 10.0, dim=-1)
        else:
            prototype_probs = torch.zeros_like(classifier_probs)

    lex_scores = lexical_scores(prepared_text, labels, device)
    confidence_scores = calibrated_confidence_scores(
        classifier_probs,
        prototype_probs,
        lex_scores,
        prototypes is not None,
    )
    for label_index, label in enumerate(labels):
        if label in DISPLAY_EXCLUDED_LABELS:
            confidence_scores[label_index] = -1.0

    top_k = min(top_k, confidence_scores.shape[-1])
    scores, indices = torch.topk(confidence_scores, top_k)

    rows = []
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        classifier_score = float(classifier_probs[index].item())
        prototype_score = float(prototype_probs[index].item())
        label = labels[index] if index < len(labels) else f"LABEL_{index}"
        rows.append(
            {
                "Sira": rank,
                "Semptom": label,
                "Guven (%)": round(score * 100, 2),
                "BERT (%)": round(classifier_score * 100, 2),
                "Benzerlik (%)": round(prototype_score * 100, 2),
                "Eslesme (%)": round(float(lex_scores[index].item()) * 100, 2),
                "Label ID": index,
            }
        )

    return pd.DataFrame(rows), prepared_text


def load_metadata() -> dict:
    return read_json(METADATA_PATH)


metadata = load_metadata()
training_metrics = read_json(TRAINING_METRICS_PATH)

st.title("House MD BERT Klinik Metin Analizi")
st.caption("BERT tabanli semptom siniflandirma prototipi")

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
    st.write("Model ciktisi yalnizca siniflandirma sonucudur.")

    st.divider()
    st.header("Model")
    st.write(f"Model: `{MODEL_DIR.name}`")
    st.write("Girdi: `text`")
    st.write("Hedef: `Symptom`")
    st.write("Tahmin: BERT + prototip benzerligi + ifade kalibrasyonu")

    if training_metrics:
        st.metric("Egitim satiri", training_metrics.get("row_count", "-"))
        st.metric("Egitim sinifi", training_metrics.get("class_count", "-"))
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
            st.subheader("BERT Destekli Tahminler")
            st.dataframe(
                predictions,
                hide_index=True,
                use_container_width=True,
            )

            chart_data = predictions.set_index("Semptom")["Guven (%)"]
            st.bar_chart(chart_data)

            top_score = float(predictions.iloc[0]["Guven (%)"])
            if top_score < 35:
                st.warning(
                    "Model guveni dusuk. Bu cikti yalnizca proje ici siniflandirma sinyali olarak degerlendirilmelidir."
                )

            with st.expander("BERT girdisi"):
                st.code(prepared_text, language="text")
