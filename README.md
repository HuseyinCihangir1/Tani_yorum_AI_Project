# Tani Yorum AI Project

House MD veri setiyle hazirlanan BERT tabanli semptom siniflandirma projesi.

## Calistirma

```powershell
.\.venv\Scripts\python.exe -X utf8 src\preprocess.py
.\.venv\Scripts\python.exe -X utf8 src\train_bert.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

GPU ile egitim icin bu projede CUDA 12.8 destekli PyTorch kullanildi:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Streamlit arayuzu mevcut model adaylari arasindan test weighted-F1 degeri daha
iyi olan modeli secer. Adaylar:

- `models/house_bert_model_finetuned`
- `models/house_bert_model_improved_augmented`
- `models/house_bert_model_improved`
- `models/house_bert_model`

Tahmin ekrani yalnizca BERT siniflandirma skorunu gosterir. `BERT (%)` degeri,
modelin softmax cikisindan gelen dogrudan sinif olasiligidir; prototip
benzerligi veya ifade kalibrasyonu kullanilmaz.

Egitimde hedef sinif alani top 30 semptom + `diger` olarak modellenir. Top 30
disinda kalan semptom satirlari atilmaz; `diger` sinifina map edilerek egitime
dahil edilir. Son fine-tuned model 4941 islenmis satir ve 31 cikis sinifi ile
egitilmistir.

`src/train_bert.py` CUDA destekli PyTorch bulursa encoder dahil tam BERT
fine-tuning yapar ve sonucu `models/house_bert_model_finetuned` altina kaydeder.
CUDA yoksa CPU fallback modunda yalnizca siniflandirma kafasi egitilir; bu mod
proje demosu icin calisir ama `BERT (%)` degerini ciddi seviyede artirmasi
beklenmez.

## Etik Sinirlar

Bu uygulama egitim ve proje gosterimi icindir. Tani koymaz, tedavi onermez ve
doktor degerlendirmesinin yerine gecmez. Acil durumlarda 112 veya en yakin acil
servis esas alinmalidir.
