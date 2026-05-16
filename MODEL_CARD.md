# Model Card - House MD Klinik Muhakeme Prototipi

## Modelin Amaci

Bu proje, House MD veri setindeki kurgusal klinik repliklerden metin oruntuleri ogrenerek iki egitim amacli NLP gorevi yapar:

- Klinik sinyal haritasi: metni en sik semptom etiketleriyle iliskilendirir.
- Iletisim tonu analizi: metni en sik duygu/ton etiketleriyle iliskilendirir.

Model tibbi tani koymak, tedavi onermek, acil durum triyaji yapmak veya klinik karar vermek icin tasarlanmamistir.

## Model Turu

- Ozellikler: kelime TF-IDF ve karakter TF-IDF
- Siniflandirici: LinearSVC
- Veri bolme: %80 egitim, %20 test
- Semptom gorevi: en sik 50 sinif
- Duygu gorevi: en sik 10 sinif

## Performans Ozeti

Semptom modeli:

- Test satiri: 158
- Accuracy: 0.709
- Macro F1: 0.617
- Weighted F1: 0.684
- Top-3 Accuracy: 0.848

Duygu modeli:

- Test satiri: 653
- Accuracy: 0.542
- Macro F1: 0.283
- Weighted F1: 0.520
- Top-3 Accuracy: 0.815

## Uygun Kullanim

- Ders projesi demonstrasyonu
- NLP on isleme ve siniflandirma ornegi
- Kurgusal veri uzerinden klinik metin analizi fikrini gostermek
- Sorumlu yapay zeka ve etik sinir tartismasi

## Uygun Olmayan Kullanim

- Gercek hasta tanisi
- Tedavi, ilac veya test onerisi
- Acil durum karari
- Hekim veya uzman gorusu yerine kullanma
- Gercek hasta verisiyle izinsiz kullanim
- Klinik ortamda dogrulanmadan karar destek sistemi olarak kullanma

## Veri Kaynagi ve Sinirlar

Veri seti House MD dizisi gibi kurgusal bir kaynaktan olusturulmustur. Bu nedenle:

- Gercek hasta dagilimini temsil etmez.
- Dramatik anlatim ve senaryo kaynakli yanliliklar icerebilir.
- Nadir veya karmasik durumlar abartili temsil edilmis olabilir.
- Bazi etiketlerde ornek sayisi cok azdir.
- Model gercek klinik genelleme iddiasi tasimaz.

## Bilinen Riskler

- Yanlis semptom sinyali uretme
- Dusuk ornekli siniflarda zayif performans
- Kurgusal veri kaynakli yanlilik
- Siralama payinin gercek olasilik gibi yanlis yorumlanmasi
- Kullanici tarafindan tibbi tavsiye gibi algilanma
- Kisisel saglik verisi girilmesi halinde gizlilik riski

## Risk Azaltma Onlemleri

- Uygulamada etik onay kutusu zorunlu hale getirildi.
- Cikti basliklari "tani" yerine "klinik sinyal" ve "muhakeme ipucu" olarak degistirildi.
- Acil durum ifadeleri icin 112 / acil saglik destegi uyarisi eklendi.
- E-posta, telefon ve TC kimlik benzeri veri icin kisisel veri uyarisi eklendi.
- Sonuclar "siralama payi" olarak gosteriliyor; klinik guven skoru olarak sunulmuyor.
- Uygulama icinde kurgusal veri ve egitim amaci acikca belirtiliyor.

## Insan Denetimi

Bu sistemin ciktisi tek basina kullanilmamalidir. Gercek klinik senaryolarda her sonuc yetkili saglik profesyoneli tarafindan degerlendirilmelidir.

## Etik ve Mevzuat Referanslari

- WHO, saglikta yapay zeka icin guvenlik, seffaflik, hesap verebilirlik ve insan denetimi ilkelerini vurgular: https://www.who.int/news/item/16-05-2023-who-calls-for-safe-and-ethical-ai-for-health
- Avrupa Komisyonu, saglikta yapay zeka uygulamalarinda guvenlik, kalite, insan gozetimi ve risk yonetimi ihtiyacini vurgular: https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en
- KVKK kapsaminda saglik verileri ozel nitelikli kisisel veri olarak degerlendirilir: https://www.kvkk.gov.tr/Icerik/2051/Ozel-Nitelikli-Kisisel-Veriler
