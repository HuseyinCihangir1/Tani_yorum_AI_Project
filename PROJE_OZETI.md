# Proje Ozeti

Bu proje, tibbi metinlerden ayirici tani / semptom siniflandirmasi yapmayi hedefleyen BERT tabanli bir NLP calismasidir. README dosyasinda hedef olarak fine-tune edilmis BERT modeli, SHAP ile aciklanabilir yapay zeka destegi ve Streamlit web arayuzu belirtilmistir.

## Su Ana Kadar Yapilanlar

- Proje icin temel Python bagimliliklari `requirements.txt` dosyasinda listelenmistir.
- Veri on isleme icin `src/preprocess.py` dosyasi hazirlanmistir.
- Ham veri olarak `data/raw/house_md_data.csv` dosyasinin okunmasi planlanmistir.
- On isleme adimlarinda metinlerin kucuk harfe cevrilmesi, noktalama isaretlerinin temizlenmesi ve `clean_text` alaninin uretilmesi hedeflenmistir.
- `Symptom` sutunundan sayisal `label` degerleri olusturulacak sekilde etiketleme mantigi eklenmistir.
- Islenen verinin `data/processed/ready_data.csv` konumuna UTF-8 formatinda kaydedilmesi planlanmistir.
- BERT egitimi icin `src/train_bert.py` dosyasi olusturulmustur.
- Model olarak `dbmdz/bert-base-turkish-cased` Turkce BERT modeli secilmistir.
- Egitimi hizlandirmak icin bos metinleri temizleme, en populer 100 sinifi secme ve etiketleri yeniden siralama adimlari eklenmistir.
- Hugging Face `datasets`, `transformers`, `Trainer` ve PyTorch kullanilarak egitim akisi kurulmustur.
- GPU varsa CUDA kullanimi, fp16 egitim, batch size ve tek epoch gibi hiz odakli egitim ayarlari yapilmistir.
- Egitilen modelin `models/house_bert_model` klasorune kaydedilmesi planlanmistir.

## Henuz Repoda Gorunmeyen / Tamamlanacak Kisimlar

- `data` klasoru ve ornek/ham veri dosyalari repoda gorunmemektedir.
- Egitilmis model dosyalari repoda bulunmamaktadir.
- README'de belirtilen SHAP aciklanabilirlik kisminin kodu henuz eklenmemistir.
- README'de belirtilen Streamlit web arayuzu henuz repoda gorunmemektedir.
- Degerlendirme metrikleri, test sonuclari veya model performans raporu henuz eklenmemistir.

## Genel Durum

Proje, veri on isleme ve Turkce BERT fine-tuning hattinin temelini atmis durumda. Bir sonraki mantikli adimlar; veri dosyasini eklemek, on isleme scriptini calistirip islenmis veriyi uretmek, modeli egitmek, performansi raporlamak ve ardindan SHAP ile yorumlanabilirlik ve Streamlit arayuzunu tamamlamaktir.
