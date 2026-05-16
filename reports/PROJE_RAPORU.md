# House MD Tani Yorum AI Projesi - Yazili Rapor Taslagi

## 1. House MD Veri Setinin Hazirlanmasi Asamasi

Bu projede House MD dizisinden elde edilen replik tabanli tibbi metin veri seti kullanilmistir. Veri seti 15 Nisan 2026 teslim asamasi icin CSV formatinda hazirlanmis ve proje icinde `data/raw/house_md_data.csv` konumuna yerlestirilmistir.

Ham veri seti 7282 satir ve 16 sutundan olusmaktadir. Temel sutunlar `text`, `Symptom`, `Emotion`, `Organ`, `Intent`, `diagnosis_stage`, `speaker`, `season` ve `episode` alanlaridir. Bu calismada ana metin girdisi olarak `text`, semptom/tani siniflandirmasi icin `Symptom`, duygu analizi icin `Emotion` sutunu kullanilmistir.

## 2. On Isleme ve Ozellik Secimi Asamasi

On isleme islemleri `src/preprocess.py` dosyasinda gerceklestirilmistir. CSV dosyasi UTF-8 agirlikli okunacak sekilde ayarlanmis, ayirac otomatik algilanmis ve eksik zorunlu sutun kontrolu eklenmistir.

Uygulanan islemler:

- Bos `text` ve `Symptom` degerleri temizlendi.
- `Symptom` etiketleri kucuk harfe cevrildi ve bastaki/sondaki bosluklar kaldirildi.
- `-`, `none`, `symptom`, `nan`, `belirsiz`, `yok` gibi model icin anlamsiz etiketler cikartildi.
- Metinler kucuk harfe cevrildi, noktalama isaretleri temizlendi ve `clean_text` sutunu olusturuldu.
- Siniflandirma icin sayisal `label` sutunu uretildi.

On isleme sonucunda veri seti 7282 satirdan 4068 satira indirilmistir. Benzersiz semptom sinifi sayisi 2562 olarak bulunmustur. Siniflar cok dengesiz oldugu icin modelleme asamasinda semptom gorevi icin en sik 50 sinif, duygu analizi icin en sik 10 duygu sinifi secilmistir.

Ozellik cikarimi icin TF-IDF yontemi kullanilmistir:

- Kelime tabanli TF-IDF: 1-2 gram
- Karakter tabanli TF-IDF: 3-5 gram

Karakter tabanli ozellikler Turkce ekler, yazim farklari ve tibbi terim varyasyonlari icin daha dayanikli temsil saglamak amaciyla eklenmistir.

## 3. Modelleme Asamasi

Proje icin iki ayri baseline model egitilmistir:

- Semptom / tani adayi siniflandirma modeli
- Duygu analizi modeli

Modelleme `src/train_baseline.py` dosyasi ile yapilmistir. Model olarak TF-IDF ozellikleri uzerinde `LinearSVC` kullanilmistir. Sinif dengesizligini azaltmak icin `class_weight="balanced"` tercih edilmistir. Veri seti %80 egitim, %20 test olacak sekilde ve sinif dagilimi korunarak ayrilmistir.

Semptom siniflandirma sonucu:

- Kullanilan satir sayisi: 790
- Sinif sayisi: 50
- Accuracy: 0.709
- Macro F1: 0.617
- Weighted F1: 0.684
- Top-3 Accuracy: 0.848

Duygu analizi sonucu:

- Kullanilan satir sayisi: 3262
- Sinif sayisi: 10
- Accuracy: 0.542
- Macro F1: 0.283
- Weighted F1: 0.520
- Top-3 Accuracy: 0.815

Sonuclar `reports/metrics/` klasorunde JSON ve siniflandirma raporu olarak saklanmistir. Egitilen modeller `models/symptom_classifier.joblib` ve `models/emotion_classifier.joblib` dosyalarina kaydedilmistir.

## 4. Modelin Canliya Alinmasi ve Etik Konusu

Modelin kullanilabilir bir bot arayuzu haline getirilmesi icin `app.py` dosyasinda Streamlit tabanli web arayuzu hazirlanmistir. Kullanici kisisel veri icermeyen klinik metin girdiginde sistem kesin tani uretmez; metindeki klinik sinyal oruntulerini ve iletisim tonu ipuclarini listeler.

Canli uygulama komutu:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Yerel calisma adresi:

```text
http://localhost:8501
```

Etik sinirlar:

- Uygulama egitim ve proje amaclidir.
- Gercek tibbi tani veya tedavi onerisi uretmez.
- Model ciktilari klinik karar yerine gecmez.
- House MD dizisi kurgusal bir kaynak oldugu icin veri gercek hasta dagilimini temsil etmeyebilir.
- Model yanli, eksik veya hatali tahminler uretebilir.
- Gercek saglik uygulamalarinda hekim onayi, klinik validasyon, veri gizliligi ve mevzuat uyumu zorunludur.

Uygulamaya eklenen etik guvenlik katmanlari:

- Analiz butonu kullanici etik kullanim kosullarini kabul etmeden aktif olmaz.
- "Tani" dili yerine "Klinik Sinyal Haritasi" dili kullanilir.
- "Duygu Analizi" dili yerine "Iletisim Tonu ve Muhakeme Ipuclari" dili kullanilir.
- Acil durum ifadeleri yakalanirsa 112 / acil saglik destegi uyarisi verilir.
- E-posta, telefon ve TC kimlik benzeri kisisel veri desenleri yakalanirsa gizlilik uyarisi verilir.
- Model skoru "olasilik" olarak degil, "siralama payi" olarak gosterilir.

Etik dokumantasyon:

- `MODEL_CARD.md`
- `reports/ETHICS_AND_GOVERNANCE.md`
- `reports/RISK_REGISTER.md`
- `reports/DATA_STATEMENT.md`

## 5. Yazili Rapor ve Sozlu Sunum

Bu dosya yazili rapor taslagi olarak kullanilabilir. Sozlu sunumda proje akisi su sirayla anlatilabilir:

1. Veri setinin amaci ve kolonlari
2. On isleme adimlari
3. Sinif dengesizligi problemi ve sinif secimi
4. TF-IDF ozellik cikarimi
5. Semptom siniflandirma ve duygu analizi modelleri
6. Model performans sonuclari
7. Streamlit bot arayuzu
8. Etik sinirlar ve gelistirme onerileri

Gelistirme onerileri:

- Daha dengeli ve daha buyuk tibbi veri seti ile egitim
- BERT veya Turkce klinik dil modeli ile karsilastirma
- SHAP veya benzeri aciklanabilir yapay zeka yontemleri
- Daha ayrintili hata analizi
- Kullanici geri bildirimi ile model iyilestirme
