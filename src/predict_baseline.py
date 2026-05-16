import argparse
from pathlib import Path

import joblib
import numpy as np

try:
    from src.preprocess import clean_text
except ModuleNotFoundError:
    from preprocess import clean_text


MODEL_DIR = Path("models")


def load_model(model_name):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model bulunamadi: {path}. Once src/train_baseline.py calistirin.")
    return joblib.load(path)


def predict_top_k(model, text, top_k=5):
    cleaned = clean_text(text)
    scores = model.decision_function([cleaned])
    scores = np.asarray(scores)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    classes = model.classes_
    top_indices = np.argsort(scores[0])[::-1][:top_k]
    top_scores = scores[0][top_indices]
    exp_scores = np.exp(top_scores - np.max(top_scores))
    confidence = exp_scores / exp_scores.sum()

    return [
        {
            "label": str(classes[index]),
            "score": float(top_scores[pos]),
            "confidence": float(confidence[pos]),
        }
        for pos, index in enumerate(top_indices)
    ]


def main():
    parser = argparse.ArgumentParser(description="House MD baseline tahmin araci")
    parser.add_argument("text", help="Tahmin edilecek klinik metin")
    parser.add_argument("--model", default="symptom_classifier", help="Model adi")
    parser.add_argument("--top-k", type=int, default=5, help="Gosterilecek tahmin sayisi")
    args = parser.parse_args()

    model = load_model(args.model)
    predictions = predict_top_k(model, args.text, args.top_k)

    for item in predictions:
        print(f"{item['label']}\t{item['confidence']:.3f}\t{item['score']:.3f}")


if __name__ == "__main__":
    main()
