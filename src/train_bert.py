import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
TARGET_COLUMN = "Symptom"
TEXT_COLUMN = "text"

DATA_PATH = Path("data/processed/ready_data.csv")
PREPROCESS_METADATA_PATH = Path("data/processed/preprocess_metadata.json")
SOURCE_MODEL_DIR = Path("models/house_bert_model")
OUTPUT_MODEL_DIR = Path("models/house_bert_model_finetuned")
OUTPUT_DIR = Path("outputs/bert_training")
EXPECTED_DATA_PIPELINE_VERSION = "symptom_exploded_canonical_v3"

TOP_N_LABELS = 30
OTHER_LABEL = "diğer"
USE_OTHER_LABEL = True
MAX_LENGTH = 128
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
FULL_FINE_TUNE_EPOCHS = 8
CLASSIFIER_HEAD_EPOCHS = 140
FULL_FINE_TUNE_PATIENCE = 3
CLASSIFIER_HEAD_PATIENCE = 20
FULL_FINE_TUNE_BATCH_SIZE = 4
CLASSIFIER_TEXT_BATCH_SIZE = 32
FEATURE_BATCH_SIZE = 128
FULL_FINE_TUNE_LR = 2e-5
CLASSIFIER_HEAD_LR = 8e-4
USE_SYNTHETIC_AUGMENTATION = True
USE_TEMPLATE_AUGMENTATION = False

SYNTHETIC_SENTENCE_TEMPLATES = (
    "Hasta {term} sikayetiyle degerlendiriliyor.",
    "Hasta {term} şikayetiyle değerlendiriliyor.",
    "Hastada {term} var.",
    "Klinik notlarda {term} belirtisi geciyor.",
    "Klinik notlarda {term} belirtisi geçiyor.",
    "Doktor hastada {term} oldugunu belirtti.",
    "Doktor hastada {term} olduğunu belirtti.",
    "Muayenede {term} bulgusu kaydedildi.",
    "{term} nedeniyle hasta takip ediliyor.",
)

ALIASES_BY_LABEL = {
    "alerjik reaksiyon": ("alerjik reaksiyon", "alerji", "anafilaksi"),
    "aritmi": ("aritmi", "ritim bozuklugu", "kalp ritmi bozuklugu"),
    "ateş": ("ates", "ateş", "atesli", "ateşli", "yuksek ates", "yüksek ateş"),
    "ağrı": ("agri", "ağrı", "siddetli agri", "şiddetli ağrı", "vucut agrisi"),
    "baş ağrısı": ("bas agrisi", "baş ağrısı", "basim agriyor", "başım ağrıyor", "migren"),
    "burun kanaması": ("burun kanamasi", "burun kanaması", "burnu kaniyor", "burnu kanıyor"),
    "böbrek yetmezliği": ("bobrek yetmezligi", "böbrek yetmezliği", "bobrek fonksiyon kaybi"),
    "depresyon": ("depresyon", "depresif durum", "mutsuzluk"),
    "döküntü": ("dokuntu", "döküntü", "cilt dokuntusu", "cilt döküntüsü", "deride dokuntu"),
    "enfeksiyon": ("enfeksiyon", "iltihap", "bakteriyel enfeksiyon"),
    "felç": ("felc", "felç", "inme", "paralizi"),
    "halüsinasyon": ("halusinasyon", "halüsinasyon", "gercek disi sesler", "sanri"),
    "kalp durması": ("kalp durmasi", "kalp durması", "kalbi durdu", "kardiyak arrest"),
    "kalp krizi": ("kalp krizi", "miyokard enfarktusu", "miyokard enfarktüsü", "enfarktus", "enfarktüs"),
    "kanama": ("kanama", "kanamasi", "kan kaybi"),
    "kanser": ("kanser", "malignite", "kotu huylu hastalik"),
    "karaciğer yetmezliği": ("karaciger yetmezligi", "karaciğer yetmezliği", "karaciger fonksiyon kaybi"),
    "karın ağrısı": ("karin agrisi", "karın ağrısı", "karnim agriyor", "karnım ağrıyor", "abdominal agri"),
    "koma": ("koma", "bilinc kapali", "bilinci kapalı", "bilinc kaybi"),
    "lezyon": ("lezyon", "doku lezyonu", "mr lezyonu"),
    "lupus": ("lupus", "sle", "sistemik lupus"),
    "nefes darlığı": (
        "nefes darligi",
        "nefes darlığı",
        "solunum sikintisi",
        "solunum sıkıntısı",
        "solunumunda rahatsizlik",
        "solunumunda rahatsızlık",
        "solumunda rahatsizlik",
        "solumunda rahatsızlık",
        "nefes alamama",
    ),
    "nöbet": ("nobet", "nöbet", "nobet gecirdi", "nöbet geçirdi", "epileptik nobet"),
    "pıhtı": ("pihti", "pıhtı", "kan pihtisi", "kan pıhtısı", "trombus"),
    "tanı": ("tani", "tanı", "teshis", "teşhis"),
    "taşikardi": ("tasikardi", "taşikardi", "hizli nabiz", "hızlı nabız", "kalp hizi yuksek"),
    "tümör": ("tumor", "tümör", "kitle", "neoplazi"),
    "vaskülit": ("vaskulit", "vaskülit", "damar iltihabi"),
    "zehirlenme": ("zehirlenme", "toksik etki", "ilac zehirlenmesi"),
    "öksürük": ("oksuruk", "öksürük", "oksuruyor", "öksürüyor", "oksurme", "öksürme"),
}

SPECIFIC_SYNTHETIC_SENTENCES = {
    "ateş": (
        "Hasta atesli.",
        "Hasta ateşli.",
        "Hastanin yuksek atesi var.",
        "Hastanın yüksek ateşi var.",
    ),
    "kalp krizi": (
        "Hastanin kalp krizi gecirdigi kesin.",
        "Hastanın kalp krizi geçirdiği kesin.",
        "Hasta kalp krizi geciriyor.",
        "Hasta kalp krizi geçiriyor.",
        "Kalp krizi bulgulari belirgin.",
        "Kalp krizi bulguları belirgin.",
    ),
    "nefes darlığı": (
        "Hastanin solunumunda rahatsizlik var.",
        "Hastanın solunumunda rahatsızlık var.",
        "Hastanın solumunda rahatsızlık var.",
        "Hasta nefes almakta zorlaniyor.",
        "Hasta nefes almakta zorlanıyor.",
        "Nefes darligi belirgin.",
        "Nefes darlığı belirgin.",
    ),
    "öksürük": (
        "Hasta surekli oksuruyor.",
        "Oksuruk sikayeti devam ediyor.",
    ),
    "nöbet": (
        "Hasta nobet gecirdi.",
        "Nobet sonrasi bilinci dalgalaniyor.",
    ),
    "kanama": (
        "Hastanin kanamasi var.",
        "Kanama kontrol altina alinamadi.",
    ),
    "taşikardi": (
        "Hastanin nabzi cok hizli.",
        "Kalp hizi yuksek ve tasikardi dusunuluyor.",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_frame() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip() != ""].copy()
    df = df[df[TARGET_COLUMN].notna()].copy()

    class_counts = df[TARGET_COLUMN].value_counts()
    selected_labels = class_counts.nlargest(TOP_N_LABELS).index.tolist()
    df["original_target"] = df[TARGET_COLUMN]

    if USE_OTHER_LABEL:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].where(
            df[TARGET_COLUMN].isin(selected_labels),
            OTHER_LABEL,
        )
    else:
        df = df[df[TARGET_COLUMN].isin(selected_labels)].copy()

    labels = sorted(df[TARGET_COLUMN].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    df["label"] = df[TARGET_COLUMN].map(label_to_id).astype(int)

    selected_counts = df[TARGET_COLUMN].value_counts().to_dict()
    selected_counts = {label: int(count) for label, count in selected_counts.items()}
    return df.reset_index(drop=True), labels, selected_counts


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_size = TEST_SIZE + VALIDATION_SIZE
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=SEED,
        stratify=df["label"],
    )

    relative_test_size = TEST_SIZE / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_synthetic_frame(labels: list[str], label_to_id: dict[str, int]) -> pd.DataFrame:
    rows = []
    seen = set()

    for label in labels:
        if USE_OTHER_LABEL and label == OTHER_LABEL:
            continue

        terms = {label}
        terms.update(ALIASES_BY_LABEL.get(label, ()))

        if USE_TEMPLATE_AUGMENTATION:
            for term in sorted(terms):
                for template in SYNTHETIC_SENTENCE_TEMPLATES:
                    text = template.format(term=term)
                    key = (text.casefold(), label)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            TEXT_COLUMN: text,
                            TARGET_COLUMN: label,
                            "label": label_to_id[label],
                            "is_synthetic": True,
                        }
                    )

        for text in SPECIFIC_SYNTHETIC_SENTENCES.get(label, ()):
            key = (text.casefold(), label)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    TEXT_COLUMN: text,
                    TARGET_COLUMN: label,
                    "label": label_to_id[label],
                    "is_synthetic": True,
                }
            )

    return pd.DataFrame(rows)


def append_synthetic_examples(
    frame: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_synthetic"] = False
    return pd.concat([frame, synthetic_df], ignore_index=True)


class BertTextDataset(Dataset):
    def __init__(self, tokenizer, texts: list[str], labels: list[int]):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def make_loader(tokenizer, df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = BertTextDataset(
        tokenizer,
        df[TEXT_COLUMN].tolist(),
        df["label"].tolist(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def configure_trainable_parameters(model, freeze_encoder: bool) -> list[str]:
    if not freeze_encoder:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return ["all"]

    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier")
        if parameter.requires_grad:
            trainable_names.append(name)

    return trainable_names


def class_weight_tensor(train_df: pd.DataFrame, num_labels: int, device: torch.device):
    counts = np.bincount(train_df["label"].to_numpy(), minlength=num_labels)
    weights = len(train_df) / (num_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def extract_features(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    encoder = getattr(model, model.base_model_prefix)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = encoder(**inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]

            all_features.append(pooled.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features), torch.cat(all_labels)


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_classifier(dropout, classifier, loader, device, loss_fn) -> dict:
    dropout.eval()
    classifier.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []
    top3_correct = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = classifier(dropout(features))
            loss = loss_fn(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_examples += labels.size(0)

            predictions = logits.argmax(dim=-1)
            top_k = min(3, logits.shape[-1])
            top_indices = logits.topk(top_k, dim=-1).indices
            top3_correct += int((top_indices == labels.unsqueeze(1)).any(dim=1).sum().item())

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
        "top_3_accuracy": top3_correct / max(total_examples, 1),
    }


def evaluate_full_model(model, loader, device, loss_fn) -> dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []
    top3_correct = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_examples += labels.size(0)

            predictions = logits.argmax(dim=-1)
            top_k = min(3, logits.shape[-1])
            top_indices = logits.topk(top_k, dim=-1).indices
            top3_correct += int((top_indices == labels.unsqueeze(1)).any(dim=1).sum().item())

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
        "top_3_accuracy": top3_correct / max(total_examples, 1),
    }


def train_classifier_epoch(dropout, classifier, loader, device, optimizer, loss_fn) -> float:
    dropout.train()
    classifier.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(dropout(features))
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)

    return total_loss / max(total_examples, 1)


def train_full_model_epoch(model, loader, device, optimizer, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)

    return total_loss / max(total_examples, 1)


def predict_full_model(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return all_labels, all_predictions


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_preprocess_metadata() -> dict:
    if not PREPROCESS_METADATA_PATH.exists():
        return {}
    return json.loads(PREPROCESS_METADATA_PATH.read_text(encoding="utf-8"))


def save_class_prototypes(model, tokenizer, df: pd.DataFrame, labels: list[str], device) -> None:
    prototype_loader = make_loader(tokenizer, df, batch_size=32, shuffle=False)
    features, encoded_labels = extract_features(model, prototype_loader, device)
    features = torch.nn.functional.normalize(features, dim=1)

    prototypes = []
    counts = []
    for label_index in range(len(labels)):
        label_features = features[encoded_labels == label_index]
        counts.append(int(label_features.shape[0]))
        centroid = label_features.mean(dim=0)
        prototypes.append(centroid)

    prototypes = torch.nn.functional.normalize(torch.stack(prototypes), dim=1)
    torch.save(
        {
            "prototypes": prototypes,
            "labels": labels,
            "counts": counts,
            "target": TARGET_COLUMN,
            "text_column": TEXT_COLUMN,
            "top_n": TOP_N_LABELS,
            "max_length": MAX_LENGTH,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        OUTPUT_MODEL_DIR / "class_prototypes.pt",
    )


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    preprocess_metadata = read_preprocess_metadata()
    pipeline_version = preprocess_metadata.get("data_pipeline_version")
    if pipeline_version and pipeline_version != EXPECTED_DATA_PIPELINE_VERSION:
        print(
            "WARNING: preprocess pipeline version mismatch: "
            f"{pipeline_version} != {EXPECTED_DATA_PIPELINE_VERSION}",
            flush=True,
        )

    df, labels, selected_counts = load_training_frame()
    train_df, val_df, test_df = split_frame(df)

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    json_id_to_label = {str(index): label for index, label in id_to_label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_fine_tune = device.type == "cuda"
    freeze_encoder = not full_fine_tune
    max_epochs = FULL_FINE_TUNE_EPOCHS if full_fine_tune else CLASSIFIER_HEAD_EPOCHS
    early_stopping_patience = (
        FULL_FINE_TUNE_PATIENCE if full_fine_tune else CLASSIFIER_HEAD_PATIENCE
    )
    batch_size = FULL_FINE_TUNE_BATCH_SIZE if full_fine_tune else CLASSIFIER_TEXT_BATCH_SIZE
    learning_rate = FULL_FINE_TUNE_LR if full_fine_tune else CLASSIFIER_HEAD_LR
    training_mode = (
        "full_bert_fine_tuning"
        if full_fine_tune
        else "frozen_bert_feature_extraction_classifier_head"
    )

    original_train_rows = len(train_df)
    synthetic_df = (
        build_synthetic_frame(labels, label_to_id)
        if USE_SYNTHETIC_AUGMENTATION
        else pd.DataFrame(columns=[TEXT_COLUMN, TARGET_COLUMN, "label", "is_synthetic"])
    )
    if USE_SYNTHETIC_AUGMENTATION:
        train_df = append_synthetic_examples(train_df, synthetic_df)
        prototype_df = append_synthetic_examples(df, synthetic_df)
    else:
        train_df = train_df.copy()
        train_df["is_synthetic"] = False
        prototype_df = df.copy()

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        SOURCE_MODEL_DIR,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.config.problem_type = "single_label_classification"
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    trainable_names = configure_trainable_parameters(model, freeze_encoder)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weight_tensor(train_df, len(labels), device)
    )

    print(f"Device: {device}", flush=True)
    print(f"Freeze encoder: {freeze_encoder}", flush=True)
    print(f"Training mode: {training_mode}", flush=True)
    print(
        "Rows: "
        f"train={len(train_df)} "
        f"(original={original_train_rows}, synthetic={len(synthetic_df)}), "
        f"val={len(val_df)}, test={len(test_df)}",
        flush=True,
    )
    print(f"Classes: {len(labels)}", flush=True)
    print(f"Trainable parameter groups: {len(trainable_names)}", flush=True)

    history = []
    best_state = None
    best_weighted_f1 = -1.0
    epochs_without_improvement = 0

    if full_fine_tune:
        train_loader = make_loader(tokenizer, train_df, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(tokenizer, val_df, batch_size=batch_size, shuffle=False)
        test_loader = make_loader(tokenizer, test_df, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=0.01,
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = train_full_model_epoch(
                model,
                train_loader,
                device,
                optimizer,
                loss_fn,
            )
            val_metrics = evaluate_full_model(model, val_loader, device, loss_fn)
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = train_loss
            history.append(val_metrics)

            print(
                "Epoch "
                f"{epoch}/{max_epochs} - train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
                f"val_top3={val_metrics['top_3_accuracy']:.4f}",
                flush=True,
            )

            if val_metrics["weighted_f1"] > best_weighted_f1:
                best_weighted_f1 = val_metrics["weighted_f1"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate_full_model(model, test_loader, device, loss_fn)
        test_true, test_predictions = predict_full_model(model, test_loader, device)
    else:
        print("Extracting frozen BERT features...", flush=True)
        train_text_loader = make_loader(
            tokenizer,
            train_df,
            batch_size=batch_size,
            shuffle=False,
        )
        val_text_loader = make_loader(tokenizer, val_df, batch_size=batch_size, shuffle=False)
        test_text_loader = make_loader(tokenizer, test_df, batch_size=batch_size, shuffle=False)

        train_features, train_labels = extract_features(model, train_text_loader, device)
        val_features, val_labels = extract_features(model, val_text_loader, device)
        test_features, test_labels = extract_features(model, test_text_loader, device)

        train_loader = make_feature_loader(
            train_features,
            train_labels,
            batch_size=FEATURE_BATCH_SIZE,
            shuffle=True,
        )
        val_loader = make_feature_loader(
            val_features,
            val_labels,
            batch_size=FEATURE_BATCH_SIZE,
            shuffle=False,
        )
        test_loader = make_feature_loader(
            test_features,
            test_labels,
            batch_size=FEATURE_BATCH_SIZE,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)

        for epoch in range(1, max_epochs + 1):
            train_loss = train_classifier_epoch(
                model.dropout,
                model.classifier,
                train_loader,
                device,
                optimizer,
                loss_fn,
            )
            val_metrics = evaluate_classifier(
                model.dropout,
                model.classifier,
                val_loader,
                device,
                loss_fn,
            )
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = train_loss
            history.append(val_metrics)

            print(
                "Epoch "
                f"{epoch}/{max_epochs} - train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
                f"val_top3={val_metrics['top_3_accuracy']:.4f}",
                flush=True,
            )

            if val_metrics["weighted_f1"] > best_weighted_f1:
                best_weighted_f1 = val_metrics["weighted_f1"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.classifier.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

        if best_state is not None:
            model.classifier.load_state_dict(best_state)

        test_metrics = evaluate_classifier(
            model.dropout,
            model.classifier,
            test_loader,
            device,
            loss_fn,
        )

        test_predictions = []
        test_true = test_labels.numpy().tolist()
        model.dropout.eval()
        model.classifier.eval()
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                logits = model.classifier(model.dropout(features))
                test_predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    report = classification_report(
        test_true,
        test_predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    save_class_prototypes(model, tokenizer, prototype_df, labels, device)

    label_mapping = {
        "target": TARGET_COLUMN,
        "text_column": TEXT_COLUMN,
        "top_n": TOP_N_LABELS,
        "other_label": OTHER_LABEL if USE_OTHER_LABEL else None,
        "uses_other_label": USE_OTHER_LABEL,
        "class_count": len(labels),
        "row_count": int(len(df)),
        "label_to_id": label_to_id,
        "id_to_label": json_id_to_label,
        "class_counts": selected_counts,
    }
    save_json(OUTPUT_MODEL_DIR / "label_mapping.json", label_mapping)
    save_json(Path("data/processed/bert_trained_symptom_label_mapping.json"), label_mapping)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_dir": str(SOURCE_MODEL_DIR),
        "output_model_dir": str(OUTPUT_MODEL_DIR),
        "data_pipeline_version": pipeline_version,
        "expected_data_pipeline_version": EXPECTED_DATA_PIPELINE_VERSION,
        "preprocess_created_at_utc": preprocess_metadata.get("created_at_utc"),
        "seed": SEED,
        "device": str(device),
        "freeze_encoder": freeze_encoder,
        "top_n_labels": TOP_N_LABELS,
        "other_label": OTHER_LABEL if USE_OTHER_LABEL else None,
        "uses_other_label": USE_OTHER_LABEL,
        "max_length": MAX_LENGTH,
        "epochs": len(history),
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size,
        "feature_batch_size": FEATURE_BATCH_SIZE if not full_fine_tune else None,
        "learning_rate": learning_rate,
        "training_mode": training_mode,
        "split_rows": {
            "train": int(len(train_df)),
            "train_original": int(original_train_rows),
            "train_synthetic": int(len(synthetic_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "other_class_rows": int(selected_counts.get(OTHER_LABEL, 0)),
        "augmentation": {
            "enabled": USE_SYNTHETIC_AUGMENTATION,
            "synthetic_rows": int(len(synthetic_df)),
            "template_augmentation_enabled": USE_TEMPLATE_AUGMENTATION,
            "templates_per_term": len(SYNTHETIC_SENTENCE_TEMPLATES),
            "specific_sentence_labels": sorted(SPECIFIC_SYNTHETIC_SENTENCES),
        },
        "class_count": len(labels),
        "row_count": int(len(df)),
        "history": history,
        "best_validation_weighted_f1": best_weighted_f1,
        "test_metrics": test_metrics,
        "classification_report": report,
    }
    save_json(OUTPUT_MODEL_DIR / "training_metrics.json", metrics)
    save_json(OUTPUT_DIR / "training_metrics.json", metrics)

    print("Training complete.", flush=True)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Model saved to: {OUTPUT_MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
