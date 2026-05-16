# House MD Tani Yorum AI Project

House MD veri seti uzerinden tibbi metinlerde klinik sinyal haritasi ve iletisim tonu ipuclari cikaran egitim amacli NLP projesi.

## Proje Akisi

1. `data/raw/house_md_data.csv` ham veri dosyasi eklenir.
2. `src/preprocess.py` ile temizlenmis veri uretilir.
3. `src/train_baseline.py` ile semptom ve duygu modelleri egitilir.
4. `app.py` ile etik guvenlik katmanli Streamlit arayuzu calistirilir.
5. Yazili rapor, model karti, veri beyani ve risk kaydi incelenir.

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

## Veri On Isleme

```powershell
.\.venv\Scripts\python.exe src\preprocess.py
```

Cikti:

```text
data/processed/ready_data.csv
```

## Model Egitimi

```powershell
.\.venv\Scripts\python.exe -X utf8 src\train_baseline.py
```

Ciktilar:

```text
models/symptom_classifier.joblib
models/emotion_classifier.joblib
reports/metrics/
```

## Tahmin Ornegi

```powershell
.\.venv\Scripts\python.exe -X utf8 src\predict_baseline.py "Hasta ates, oksuruk ve nefes darligi yasiyor"
```

## Canli Arayuz

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Yerel adres:

```text
http://localhost:8501
```

## Etik ve Guvenlik

Bu proje egitim amaclidir. Uretilen tahminler tibbi tani, tedavi veya acil karar yerine kullanilamaz. Gercek klinik kullanim icin uzman dogrulamasi, veri gizliligi, mevzuat uyumu ve kapsamli validasyon gerekir.

Uygulamaya eklenen etik onlemler:

- Kullanici etik kullanim onayi vermeden analiz baslamaz.
- Ciktilar "tani" yerine "klinik sinyal" olarak sunulur.
- Acil durum ifadelerinde 112 / acil saglik destegi uyarisi verilir.
- E-posta, telefon ve TC kimlik benzeri veri icin kisisel veri uyarisi verilir.
- Model skoru klinik olasilik degil, siralama payi olarak gosterilir.

Etik dokumanlar:

```text
MODEL_CARD.md
reports/ETHICS_AND_GOVERNANCE.md
reports/RISK_REGISTER.md
reports/DATA_STATEMENT.md
```

## Sunum Briefi

Farkli bir AI ile sunum hazirlamak icin proje ozeti ve slayt akisi:

```text
reports/presentation/SUNUM_HAZIRLAMA_BRIEF.md
```
