# Veri Beyani

## Veri Kaynagi

Projede `data/raw/house_md_data.csv` dosyasi kullanilmistir. Veri seti House MD dizisi baglaminda hazirlanmis kurgusal replik ve etiketlerden olusur.

## Ham Veri Ozeti

- Satir sayisi: 7282
- Sutun sayisi: 16
- Temel sutunlar: `text`, `Symptom`, `Emotion`, `Organ`, `Intent`, `diagnosis_stage`, `speaker`, `season`, `episode`

## Islenmis Veri Ozeti

- Satir sayisi: 4068
- Sutun sayisi: 18
- Eklenen sutunlar: `clean_text`, `label`
- Benzersiz semptom etiketi: 2562

## Temizleme Kararlari

- Bos metin ve bos semptom alanlari cikartildi.
- `-`, `none`, `symptom`, `nan`, `belirsiz`, `yok` gibi anlamsiz etiketler cikartildi.
- Etiketler kucuk harfe cevrildi.
- Metinden noktalama isaretleri kaldirildi.
- Coklu bosluklar tek bosluga indirildi.

## Modelleme Icin Secilen Veri

Semptom gorevinde veri cok dengesiz oldugu icin en sik 50 sinif kullanildi. Bu alt kumede 790 satir yer aldi.

Duygu gorevinde en sik 10 sinif kullanildi. Bu alt kumede 3262 satir yer aldi.

## Veri Etigi ve Gizlilik

Veri kurgusal oldugu icin gercek hasta kimligi icermesi beklenmez. Buna ragmen proje uygulamasinda kullanicilarin gercek hasta adi, telefon, e-posta, TC kimlik numarasi veya benzeri kisisel veri girmemesi gerekir.

Gercek klinik veri ile calisilacaksa veri anonimlestirme, kurum izni, hukuki dayanak, erisim kontrolu ve veri minimizasyonu zorunlu degerlendirme basliklari olmalidir.

## Temsil Sinirlari

- House MD dramatik ve kurgusal bir kaynaktir.
- Nadir hastaliklar veya dramatik vakalar abartili temsil edilebilir.
- Diyaloglar gercek klinik not formatinda olmayabilir.
- Modelin gercek hasta verisine dogrudan genellenmesi beklenmemelidir.
