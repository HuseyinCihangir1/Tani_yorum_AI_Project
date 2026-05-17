# Tani Yorum AI Project

House MD veri setiyle hazirlanan BERT tabanli semptom siniflandirma projesi.

## Calistirma

```powershell
.\.venv\Scripts\python.exe -X utf8 src\preprocess.py
.\.venv\Scripts\python.exe -X utf8 src\train_bert.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit arayuzu mevcut model adaylari arasindan test weighted-F1 degeri daha
iyi olan modeli secer. Adaylar:

- `models/house_bert_model_improved_augmented`
- `models/house_bert_model_improved`
- `models/house_bert_model`

Tahmin ekrani BERT siniflandirma skoru, BERT prototip benzerligi ve klinik ifade
kalibrasyonunu birlikte gosterir. `Guven (%)` olasilik toplami degil, bu uc
sinyalin kullaniciya okunabilir hale getirilmis karar puanidir.

## Etik Sinirlar

Bu uygulama egitim ve proje gosterimi icindir. Tani koymaz, tedavi onermez ve
doktor degerlendirmesinin yerine gecmez. Acil durumlarda 112 veya en yakin acil
servis esas alinmalidir.
