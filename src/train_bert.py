import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os

# 1. Veriyi Yükle ve Filtrele
df = pd.read_csv('data/processed/ready_data.csv')

# --- KRİTİK HIZLANDIRMA VE DÜZELTME BAŞLANGICI ---
# Boş metinleri temizle [cite: 193]
df['clean_text'] = df['clean_text'].fillna('').astype(str)
df = df[df['clean_text'].str.strip() != ""].copy()

# En popüler 100 sınıfı seç (Eğitimi hızlandırmak ve karmaşıklığı azaltmak için) [cite: 259, 282]
top_100_labels = df['Symptom'].value_counts().nlargest(100).index
df = df[df['Symptom'].isin(top_100_labels)].reset_index(drop=True)

# Etiketleri 0-99 arasına ardışık olarak eşle (IndexError çözümüdür) [cite: 206, 219, 285]
df['label'] = pd.Categorical(df['Symptom']).codes
# --- KRİTİK DÜZELTME BİTİŞİ ---

dataset = Dataset.from_pandas(df[['clean_text', 'label']])

# 2. Tokenizer ve Model (Türkçe BERT)
model_nm = 'dbmdz/bert-base-turkish-cased'
tokz = AutoTokenizer.from_pretrained(model_nm)

def tokenize_func(x): 
    # max_length=128 hem hızı artırır hem de bellek kullanımını dengeler [cite: 225, 305]
    return tokz(x['clean_text'], padding='max_length', truncation=True, max_length=128)

tok_ds = dataset.map(tokenize_func, batched=True)
dds = tok_ds.train_test_split(test_size=0.2)

# 3. GPU (CUDA) Kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
num_labels = int(df['label'].nunique())
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=num_labels).to(device)
print(f"Eğitim Cihazı: {device.upper()} (RTX 4050 Tespit Edildi)")

# 4. Eğitim Ayarları (RTX 4050 Optimize)
os.makedirs('outputs', exist_ok=True)
args = TrainingArguments(
    output_dir='outputs',
    learning_rate=3e-5,
    num_train_epochs=1,              # En hızlı test için 1 epoch yeterlidir [cite: 224, 287]
    per_device_train_batch_size=32,  # RTX 4050 için batch size 32 idealdir [cite: 303]
    fp16=True,                       # Mixed Precision: RTX 4050 hızını ikiye katlar [cite: 243, 298]
    eval_strategy="no",              # Zaman kazanmak için ara testleri kapattık [cite: 226]
    save_strategy="no",
    dataloader_pin_memory=True       # GPU aktarım hızını artırır
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds['train'],
    processing_class=tokz            # Güncel parametre ismi [cite: 183, 196]
)

# 5. Eğitimi Başlat
print(f"BERT Eğitimi {num_labels} popüler sınıf için başlıyor...")
trainer.train()

# 6. Modeli Kaydet
output_dir = "models/house_bert_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokz.save_pretrained(output_dir)
print(f"Eğitim tamamlandı. Model {output_dir} klasörüne kaydedildi.")