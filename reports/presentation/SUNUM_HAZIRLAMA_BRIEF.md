# Sunum Hazirlama Briefi - House MD Klinik Muhakeme AI Prototipi

Bu dosya, projeyi bilmeyen bir yapay zeka aracinin veya tasarim aracinin sunum hazirlayabilmesi icin gerekli tum bilgileri icerir. Amaç; anlaşılır, gereksiz teknik ayrıntıya boğulmayan, akademik olarak düzgün ve canlı demo ile desteklenebilen bir sunum hazırlamaktır.

## 1. Sunumun Ana Mesaji

Bu proje, House MD veri seti üzerinde geliştirilen eğitim amaçlı bir NLP prototipidir. Sistem gerçek tıbbi tanı koymaz; kurgusal klinik diyaloglardan öğrenilen metin örüntülerine göre:

- Klinik sinyal haritası çıkarır.
- İletişim tonu ve muhakeme ipuçlarını gösterir.
- Sonuçları canlı Streamlit arayüzünde sunar.
- Etik sınırları ve güvenlik uyarıları olan sorumlu yapay zeka yaklaşımı kullanır.

Sunum boyunca vurgulanması gereken ana cümle:

> Bu proje bir tanı koyma sistemi değil; klinik metinlerdeki sinyalleri, iletişim tonunu ve modelin etik sınırlarını gösteren eğitim amaçlı bir yapay zeka prototipidir.

## 2. Grup Uyeleri ve Gorev Dagilimi

Grup üyeleri:

- Ömer Faruk Pehlivan - 030123083
- Hüseyin Yusuf Cihangir - 030121037
- Faruk Laçin - 030122117

Görev dağılımı:

### Hüseyin Yusuf Cihangir

Hüseyin veri hazırlama ve veri ön işleme aşamalarından sorumludur.

Sorumlulukları:

- House MD veri setinin projeye uygun formatta hazırlanması
- CSV dosyasının kolon yapısının kontrol edilmesi
- Ham veri dosyasının `data/raw/house_md_data.csv` konumuna yerleştirilmesi
- `text`, `Symptom`, `Emotion` gibi kritik kolonların kontrol edilmesi
- Eksik ve anlamsız verilerin belirlenmesi
- Ön işleme çıktısının kontrol edilmesi
- Temizlenmiş veri setinin modele hazır hale getirilmesi

### Ömer Faruk Pehlivan

Ömer Faruk proje koordinasyonu, modelleme ve canlı demo tarafında sorumludur.

Sorumlulukları:

- Proje akışının hocanın istediği 5 aşamaya göre organize edilmesi
- Semptom / klinik sinyal sınıflandırma modelinin kurulması
- Modelleme stratejisinin belirlenmesi
- Streamlit canlı arayüzünün hazırlanması
- Demo akışının planlanması
- Sunumun genel hikayesinin ve ana mesajının oluşturulması

### Faruk Laçin

Faruk Laçin duygu/iletişim tonu modeli, sonuç raporlama ve etik dokümantasyon tarafında sorumludur.

Sorumlulukları:

- İletişim tonu / duygu sınıflandırma modelinin hazırlanması
- Model performans metriklerinin raporlanması
- Sonuçların sunumda anlaşılır hale getirilmesi
- Etik ve güvenlik dokümantasyonuna katkı verilmesi
- Rapor ve sunum içeriklerinin düzenlenmesi
- Görsel tutarlılık ve sunum son kontrolleri

## 3. Sunum Tasarim Kurallari

Sunumu hazırlayacak AI aracı için temel tasarım kuralı:

- Genel arka plan rengi: `#61000a`
- Genel yazı rengi: beyaz
- Vurgu rengi olarak açık gri, beyaz veya çok açık pembe kullanılabilir.
- Slaytlar sade, akademik ve okunabilir olmalıdır.
- Gereksiz görsel kalabalık, yoğun animasyon ve uzun paragraflar kullanılmamalıdır.
- Başlıklar kısa ve net olmalıdır.
- Metrikler kart veya tablo olarak gösterilebilir.
- Etik slaytı güçlü ve net olmalıdır.

Önerilen stil:

```text
Arka plan: #61000a
Ana yazı: #FFFFFF
İkincil yazı: #F4E9EA
Vurgu/kart çizgisi: #FFFFFF veya #D9B8BC
```

## 4. Hocanin Istedigi Asamalar ve Projede Karsiligi

Hocanın verdiği başlıklar:

1. House MD veri setinin hazırlanması aşaması
2. Ön işleme ve özellik seçimi aşaması
3. Modelleme aşaması
4. Modelin canlıya alınması ve etik konusu
5. Yazılı rapor ve sözlü sunum

Projede karşılıkları:

### 1. Veri Setinin Hazirlanmasi

Ham veri dosyası:

```text
data/raw/house_md_data.csv
```

Ham veri özeti:

- Satır sayısı: 7282
- Sütun sayısı: 16
- Veri tipi: House MD dizisi bağlamında hazırlanmış kurgusal klinik diyalog/veri seti
- Ana metin kolonu: `text`
- Semptom hedef kolonu: `Symptom`
- Duygu/ton hedef kolonu: `Emotion`

Önemli kolonlar:

- `season`
- `episode`
- `speaker`
- `Symptom`
- `Test`
- `Drug`
- `Procedure`
- `Intent`
- `diagnosis_stage`
- `Sarcasm`
- `Emotion`
- `Organ`
- `correct_prediction`
- `model_prediction`
- `text`
- `medical_entities`

Sunumda veri seti için anlatılacak kısa metin:

> Veri seti House MD dizisi üzerinden hazırlanmış kurgusal tıbbi diyaloglardan oluşmaktadır. Model girdisi olarak `text`, klinik sinyal hedefi olarak `Symptom`, iletişim tonu hedefi olarak `Emotion` kullanılmıştır.

### 2. On Isleme ve Ozellik Secimi

Ön işleme dosyası:

```text
src/preprocess.py
```

Ön işleme sonucunda oluşan dosya:

```text
data/processed/ready_data.csv
```

Ön işleme adımları:

- CSV dosyası UTF-8 ağırlıklı güvenli okundu.
- Ayırıcı otomatik algılandı.
- Zorunlu kolonlar kontrol edildi: `text`, `Symptom`
- Boş `text` ve boş `Symptom` satırları çıkarıldı.
- `Symptom` etiketleri küçük harfe çevrildi.
- Baştaki ve sondaki boşluklar temizlendi.
- `-`, `none`, `symptom`, `nan`, `belirsiz`, `yok` gibi anlamsız etiketler çıkarıldı.
- Metinler küçük harfe çevrildi.
- Noktalama işaretleri temizlendi.
- Çoklu boşluklar tek boşluğa indirildi.
- `clean_text` kolonu oluşturuldu.
- Sayısal `label` kolonu oluşturuldu.

Ön işleme sonucu:

- Ham veri: 7282 satır
- İşlenmiş veri: 4068 satır
- İşlenmiş veri sütun sayısı: 18
- Benzersiz semptom etiketi: 2562

Özellik seçimi:

- Semptom sınıflandırması için en sık 50 sınıf seçildi.
- Duygu/ton sınıflandırması için en sık 10 sınıf seçildi.
- Bunun nedeni veri setindeki sınıf dengesizliğidir.

Kullanılan metin özellikleri:

- Kelime tabanlı TF-IDF: 1-2 gram
- Karakter tabanlı TF-IDF: 3-5 gram

Sunumda özellik seçimi için anlatılacak kısa metin:

> Veri setinde çok fazla ve dengesiz semptom etiketi bulunduğu için ilk modelde en sık görülen sınıflar seçildi. Metinleri sayısal hale getirmek için kelime ve karakter tabanlı TF-IDF kullanıldı.

### 3. Modelleme

Modelleme dosyası:

```text
src/train_baseline.py
```

Tahmin dosyası:

```text
src/predict_baseline.py
```

Eğitilen iki model:

1. Klinik sinyal / semptom modeli
2. İletişim tonu / duygu modeli

Kullanılan algoritma:

```text
TF-IDF + LinearSVC
```

Neden bu model seçildi?

- Hızlı eğitilir.
- Açıklanabilir ve baseline olarak savunulabilir.
- BERT gibi ağır modellere geçmeden önce güvenilir bir karşılaştırma noktası sağlar.
- Küçük ve dengesiz veri setlerinde hızlı prototipleme için uygundur.

Eğitim ayarları:

- Eğitim/test bölünmesi: %80 / %20
- Stratified split kullanıldı.
- `class_weight="balanced"` kullanıldı.
- Amaç sınıf dengesizliğinin etkisini azaltmaktı.

Model çıktıları:

```text
models/symptom_classifier.joblib
models/emotion_classifier.joblib
models/baseline_metadata.json
```

Metrik dosyaları:

```text
reports/metrics/symptom_classifier_metrics.json
reports/metrics/emotion_classifier_metrics.json
reports/metrics/symptom_classifier_classification_report.txt
reports/metrics/emotion_classifier_classification_report.txt
```

#### Semptom / Klinik Sinyal Modeli Sonuclari

- Kullanılan satır sayısı: 790
- Sınıf sayısı: 50
- Test satırı: 158
- Accuracy: 0.709
- Macro F1: 0.617
- Weighted F1: 0.684
- Top-3 Accuracy: 0.848

Yorum:

> Semptom modeli, ilk 3 aday içinde doğru sınıfa yaklaşma açısından güçlü bir baseline performansı göstermiştir. Ancak bu sonuç klinik kullanım için yeterlilik anlamına gelmez; yalnızca eğitim/prototip başarısı olarak yorumlanmalıdır.

#### Iletisim Tonu / Duygu Modeli Sonuclari

- Kullanılan satır sayısı: 3262
- Sınıf sayısı: 10
- Test satırı: 653
- Accuracy: 0.542
- Macro F1: 0.283
- Weighted F1: 0.520
- Top-3 Accuracy: 0.815

Yorum:

> Duygu/ton analizi daha bağlama dayalı olduğu için zor bir görevdir. Top-3 başarının yüksek olması, modelin çoğu durumda doğru tona yakın adayları listeleyebildiğini göstermektedir.

### 4. Canliya Alma ve Etik

Canlı arayüz dosyası:

```text
app.py
```

Kullanılan araç:

```text
Streamlit
```

Yerel demo adresi:

```text
http://localhost:8501
```

Çalıştırma komutu:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Uygulama ne yapıyor?

- Kullanıcıdan klinik metin alır.
- Etik onay kutusu işaretlenmeden analiz başlatmaz.
- Klinik sinyal haritası üretir.
- İletişim tonu ve muhakeme ipuçlarını listeler.
- Acil durum ifadelerini yakalarsa 112/acil sağlık desteği uyarısı verir.
- Telefon, e-posta, TC kimlik benzeri veri girilirse kişisel veri uyarısı verir.

Uygulamada özellikle kaçınılan dil:

- "Tanı koydu"
- "Hastalık budur"
- "Tedavi önerisi"
- "Klinik karar"

Kullanılan daha güvenli dil:

- "Klinik Sinyal Haritası"
- "İletişim Tonu ve Muhakeme İpuçları"
- "Sıralama payı"
- "Eğitim amaçlı analiz"
- "Tıbbi karar değildir"

Etik güvenlik katmanları:

- Etik onay kutusu
- Acil durum uyarısı
- Kişisel veri uyarısı
- Tanı dili yerine sinyal dili
- Model skorunun olasılık gibi sunulmaması
- Model kartı
- Veri beyanı
- Risk kaydı
- Etik ve yönetişim planı

Etik dokümanlar:

```text
MODEL_CARD.md
reports/ETHICS_AND_GOVERNANCE.md
reports/RISK_REGISTER.md
reports/DATA_STATEMENT.md
```

Sunumda etik için anlatılacak kısa metin:

> Bu projede etik konu sadece rapora yazılmadı; uygulamanın davranışına da eklendi. Sistem tanı koyduğunu iddia etmez, kullanıcıdan etik onay alır, acil durum ve kişisel veri uyarıları verir.

### 5. Yazili Rapor ve Sunum

Hazır rapor dosyaları:

```text
reports/PROJE_RAPORU.md
reports/presentation/SUNUM_HAZIRLAMA_BRIEF.md
```

Bu brief dosyası farklı bir AI ile sunum hazırlamak için ana kaynak olarak kullanılmalıdır.

## 5. Sunumda Kisa Teknik Aciklamalar

Bu bölüm, sunumda kısa bilgi kutusu veya konuşmacı notu olarak kullanılabilir. Kullanıcının sorabileceği teknik kavramlara kısa ve anlaşılır cevaplar verir.

### Benzersiz semptom etiketi ne demek?

`Symptom` sütununda kaç farklı semptom/sınıf adı olduğunu gösterir. Bizim işlenmiş verimizde 2562 farklı semptom etiketi vardır. Bu, veri setinde çok fazla sınıf olduğunu ve sınıf dengesizliği bulunduğunu gösterir.

Sunumluk cümle:

> 2562 benzersiz semptom etiketi, veri setinde 2562 farklı semptom sınıfı bulunduğunu gösterir. Bu nedenle modelleme aşamasında ölçülebilir bir baseline için en sık 50 sınıf seçilmiştir.

### Benzersiz semptomları veriden sildik mi?

Hayır. Ön işleme aşamasında benzersiz veya nadir semptomlar `ready_data.csv` dosyasından topluca silinmedi. Sadece boş, geçersiz veya anlamsız etiketler temizlendi. Ancak model eğitimi aşamasında tüm 2562 sınıf yerine en sık 50 semptom sınıfı kullanıldı.

Sunumluk cümle:

> Nadir semptomlar temiz veriden silinmedi; yalnızca ilk baseline modelin eğitim kapsamı en sık 50 sınıfla sınırlandırıldı.

### Duygu/iletişim tonu için neden 10 sınıf kullanıldı?

`Emotion` sütununda da sınıf dengesizliği vardır. Bazı duygu/ton etiketleri çok sık, bazıları ise çok az geçer. Bu yüzden ilk prototipte en sık 10 duygu/ton sınıfı seçildi.

Sunumluk cümle:

> İletişim tonu görevi bağlama daha duyarlı olduğu ve sınıflar dengesiz dağıldığı için, ilk modelde en sık 10 ton sınıfı kullanılmıştır.

### Kelime TF-IDF nedir?

Kelime TF-IDF, metindeki kelimelere önem puanı verir. Sık ama ayırt edici olmayan kelimelerin etkisini azaltır; semptom açısından önemli kelime ve kelime gruplarını öne çıkarır. Projede 1-2 kelimelik ifadeler kullanıldı.

Örnek:

```text
ateş
öksürük
nefes darlığı
karın ağrısı
```

Sunumluk cümle:

> Kelime TF-IDF, metindeki kelime ve iki kelimelik ifadeleri sayısal önem puanlarına dönüştürerek modelin öğrenmesini sağlar.

### Karakter TF-IDF nedir?

Karakter TF-IDF, kelimelere değil kelimelerin içindeki harf parçalarına bakar. Projede 3-5 karakterlik parçalar kullanıldı. Türkçe ekler, yazım farklılıkları ve tıbbi terim benzerlikleri için faydalıdır.

Örnek:

```text
kanama -> kan, ana, nam, ama, kanam
```

Sunumluk cümle:

> Karakter TF-IDF, kelimeleri harf parçalarına ayırarak Türkçe ekleri, yazım varyasyonlarını ve tıbbi terim benzerliklerini yakalamaya yardımcı olur.

### LinearSVC nedir?

LinearSVC, TF-IDF ile sayısallaştırılmış metinleri sınıflandırmak için kullanılan hızlı ve güçlü bir doğrusal makine öğrenmesi modelidir.

Sunumluk cümle:

> LinearSVC, TF-IDF özelliklerinden yola çıkarak metnin hangi sınıfa daha yakın olduğunu öğrenen doğrusal bir sınıflandırıcıdır.

### Başka model kullandık mı?

Çalışan mevcut prototipte aktif olarak kullanılan model yapısı `TF-IDF + LinearSVC` modelidir. İki ayrı görev için iki ayrı model eğitildi:

- Klinik Sinyal Modeli: `Symptom`
- İletişim Tonu Modeli: `Emotion`

Projede BERT eğitimi için dosya bulunsa da, mevcut demo ve metrikler LinearSVC baseline modelleri üzerinden üretilmiştir. BERT gelecek çalışma olarak sunulmalıdır.

Sunumluk cümle:

> Mevcut çalışan prototipte iki ayrı TF-IDF + LinearSVC baseline modeli kullanılmıştır; BERT ise gelecek geliştirme adımı olarak planlanmıştır.

### %80 eğitim / %20 test ne demek?

Verinin %80'i modelin öğrenmesi için, %20'si ise modelin daha önce görmediği örnekler üzerinde test edilmesi için ayrıldı.

Sunumluk cümle:

> Model verinin %80'iyle eğitildi, %20'lik ayrılmış test verisiyle ölçüldü.

### Stratified split nedir?

Veriyi eğitim ve test olarak bölerken sınıf dağılımını korumaya yarar. Böylece eğitim ve test setlerinde sınıfların oranları benzer kalır.

Sunumluk cümle:

> Stratified split ile sınıf dağılımının eğitim ve test setlerinde benzer kalması sağlandı.

### class_weight="balanced" ne yapar?

Sınıf dengesizliğini azaltmak için az örnekli sınıflara daha yüksek hata ağırlığı, çok örnekli sınıflara daha düşük hata ağırlığı verir. Bir anlamda hata cezalarının çarpanlarını sınıf frekansına göre otomatik ayarlar.

Basit mantık:

```text
az örnekli sınıf  -> yüksek ağırlık
çok örnekli sınıf -> düşük ağırlık
```

Sunumluk cümle:

> `class_weight="balanced"`, sınıfların örnek sayılarına göre hata cezalarını otomatik ağırlıklandırır; az örnekli sınıfların eğitimde daha fazla dikkate alınmasını sağlar.

### Accuracy nedir?

Modelin ilk tahmininin doğru olma oranıdır.

Sunumluk cümle:

> Accuracy, modelin test verisindeki ilk tahmininin ne oranda doğru olduğunu gösterir.

### Macro F1 nedir?

Her sınıfa eşit önem vererek F1 ortalaması alır. Dengesiz veri setlerinde önemlidir çünkü sık sınıfların sonucu domine etmesini azaltır.

Sunumluk cümle:

> Macro F1, her sınıfı eşit ağırlıkta değerlendirerek modelin sınıflar genelindeki dengesini gösterir.

### Weighted F1 nedir?

F1 skorunu sınıfların örnek sayılarına göre ağırlıklandırır. Çok örneği olan sınıflar skora daha fazla etki eder.

Sunumluk cümle:

> Weighted F1, sınıf büyüklüklerini dikkate alan genel F1 skorudur.

### Top-3 Accuracy nedir ve nasıl hesaplandı?

Top-3 Accuracy, gerçek etiketin modelin en yüksek skorlu ilk 3 tahmini içinde olup olmadığını ölçer. LinearSVC her sınıf için karar skoru üretir. Bu skorlar büyükten küçüğe sıralanır ve ilk 3 sınıf seçilir.

Sunumluk cümle:

> Top-3 Accuracy, doğru etiketin modelin en yüksek skorlu ilk üç önerisi içinde bulunma oranıdır. Bu proje için önemlidir çünkü sistem tek kesin tanı değil, klinik sinyal sıralaması sunar.

### Model Türkçe için mi çalışıyor?

Evet, model Türkçe veri üzerinde eğitildi. Türkçe karakterli girişler daha sağlıklı sonuç verir. Ancak `ateş` ve `ates` model için birebir aynı değildir. Karakter TF-IDF ortak harf parçalarını yakaladığı için tamamen bozulmaz ama tahminlerde fark oluşabilir.

Sunumluk cümle:

> Model Türkçe veriyle eğitilmiştir. Türkçe karakterli girişler daha sağlıklı çalışır; `ateş` ve `ates` benzer sinyal verse de model açısından tamamen aynı değildir.

### "Klinik Sinyal Haritası" ve "İletişim Tonu ve Muhakeme İpuçları" neden kullanıldı?

Eski dil olan "Semptom/Tanı Adayları" ve "Duygu Analizi" daha iddialı ve tıbbi karar veriyor gibi algılanabilir. Yeni dil, modelin sadece metinsel örüntüleri ve sinyalleri gösterdiğini daha etik şekilde ifade eder.

Sunumluk cümle:

> Çıktılar tanı veya psikolojik değerlendirme gibi sunulmasın diye "Klinik Sinyal Haritası" ve "İletişim Tonu ve Muhakeme İpuçları" dili tercih edilmiştir.

## 6. Onerilen Sunum Akisi

Sunum 10-12 slayt arası olmalıdır. Gereksiz kod ayrıntısına girilmemelidir. Her slayt sade, anlaşılır ve görsel olarak temiz olmalıdır.

### Slayt 1 - Kapak

Başlık önerisi:

```text
House MD Klinik Muhakeme AI Prototipi
```

Alt başlık:

```text
Kurgusal klinik diyaloglardan klinik sinyal, iletişim tonu ve etik güvenlik analizi
```

Grup üyeleri:

- Ömer Faruk Pehlivan - 030123083
- Hüseyin Yusuf Cihangir - 030121037
- Faruk Laçin - 030122117

### Slayt 2 - Projenin Amacı

Anlatılacaklar:

- House MD veri seti üzerinde NLP tabanlı bir prototip geliştirildi.
- Sistem tanı koymak için değil, klinik metindeki sinyalleri analiz etmek için tasarlandı.
- Canlı arayüz ve etik güvenlik katmanı eklendi.

Kısa slayt metni:

```text
Amaç: Klinik metinlerdeki semptom örüntülerini ve iletişim tonunu eğitim amaçlı analiz etmek.
```

### Slayt 3 - Ekip ve Görev Dağılımı

Üç kolonlu gösterim önerilir:

```text
Hüseyin Yusuf Cihangir
030121037
Veri hazırlama, ön işleme, temiz veri üretimi

Ömer Faruk Pehlivan
030123083
Proje koordinasyonu, klinik sinyal modeli, canlı arayüz, demo

Faruk Laçin
030122117
İletişim tonu modeli, metrik raporlama, etik dokümantasyon, sunum düzeni
```

### Slayt 4 - Veri Seti

Gösterilecek sayılar:

- 7282 ham satır
- 16 ham sütun
- 4068 işlenmiş satır
- 2562 benzersiz semptom etiketi

Önemli kolonlar:

- `text`
- `Symptom`
- `Emotion`
- `Organ`
- `Intent`
- `diagnosis_stage`

Slayt mesajı:

```text
Veri kurgusal klinik diyaloglardan oluşur; gerçek hasta verisi değildir.
```

### Slayt 5 - Ön İşleme

Akış olarak göster:

```text
CSV okuma -> Eksik veri temizliği -> Etiket temizliği -> Metin temizliği -> clean_text -> label
```

Ana sonuç:

```text
7282 satırdan 4068 temiz satıra geçildi.
```

### Slayt 6 - Özellik Seçimi

Anlatılacaklar:

- Çok fazla ve dengesiz sınıf vardı.
- Semptom modeli için en sık 50 sınıf seçildi.
- Duygu modeli için en sık 10 sınıf seçildi.
- TF-IDF kullanıldı.

Kısa teknik ifade:

```text
Kelime TF-IDF + Karakter TF-IDF
```

### Slayt 7 - Modelleme

İki model göster:

```text
Klinik Sinyal Modeli
Hedef: Symptom
Model: TF-IDF + LinearSVC

İletişim Tonu Modeli
Hedef: Emotion
Model: TF-IDF + LinearSVC
```

Ek bilgi:

```text
%80 eğitim, %20 test, stratified split, class_weight="balanced"
```

### Slayt 8 - Sonuçlar

Tablo veya kart formatı önerilir:

```text
Klinik Sinyal Modeli
Accuracy: 0.709
Macro F1: 0.617
Top-3 Accuracy: 0.848

İletişim Tonu Modeli
Accuracy: 0.542
Macro F1: 0.283
Top-3 Accuracy: 0.815
```

Sunum yorumu:

```text
Semptom modeli çalışan ve ölçülebilir bir baseline verdi. Duygu modeli daha zor bir görev olduğu için performans daha sınırlı kaldı.
```

### Slayt 9 - Canlı Arayüz

Gösterilecekler:

- Streamlit arayüzü
- Metin kutusu
- Etik onay kutusu
- Klinik sinyal haritası
- İletişim tonu ipuçları
- Acil durum / kişisel veri uyarıları

Demo adresi:

```text
http://localhost:8501
```

### Slayt 10 - Etik ve Güvenlik

En önemli slaytlardan biridir.

Madde önerileri:

- Sistem tanı koymaz.
- Tıbbi tedavi önermez.
- Gerçek hasta verisi girilmemelidir.
- Acil durumda 112 / sağlık profesyoneli önceliklidir.
- Model skoru klinik olasılık değildir.
- Kurgusal veri gerçek hasta dağılımını temsil etmez.

Etik dokümanlar:

- Model kartı
- Veri beyanı
- Risk kaydı
- Etik ve yönetişim planı

### Slayt 11 - Demo Senaryoları

Sunumda kullanılabilecek üç örnek:

Normal klinik örnek:

```text
Hasta ateş, öksürük ve nefes darlığı yaşıyor.
```

Acil durum örneği:

```text
Hasta nefes alamıyor, şiddetli kanama var.
```

Kişisel veri örneği:

```text
Hasta telefonu 05551234567, e-posta adresi test@example.com.
```

Amaç:

```text
Modelden önce etik güvenlik davranışını göstermek.
```

### Slayt 12 - Sonuç ve Gelecek Çalışmalar

Sonuç:

- Veri hazırlandı.
- Ön işleme tamamlandı.
- İki model eğitildi.
- Canlı arayüz hazırlandı.
- Etik güvenlik katmanı eklendi.
- Rapor ve sunum dokümanları oluşturuldu.

Gelecek çalışmalar:

- Tanısal Muhakeme Haritası
- SHAP ile açıklanabilirlik
- Türkçe BERT modeli
- Daha dengeli ve daha büyük veri seti
- Klinik uzman değerlendirmesi

### Slayt 13 - Teknik Kavramlar / Soru-Cevap

Bu slayt zorunlu değildir ama sunuma eklenmesi önerilir. Kısa bilgi kartları şeklinde verilebilir:

- Benzersiz semptom etiketi: `Symptom` sütunundaki farklı sınıf sayısıdır.
- Kelime TF-IDF: kelime ve kelime gruplarını sayısallaştırır.
- Karakter TF-IDF: harf parçalarını kullanarak Türkçe ek ve yazım farklarını yakalar.
- LinearSVC: TF-IDF özelliklerini sınıflandıran doğrusal modeldir.
- Top-3 Accuracy: doğru etiketin ilk üç tahmin içinde bulunma oranıdır.
- `class_weight="balanced"`: az örnekli sınıflara daha yüksek hata ağırlığı verir.

## 7. Sunum Dili ve Tonu

Sunum dili:

- Akademik ama anlaşılır
- Gereksiz kod ayrıntısı yok
- Modelin yapabilecekleri abartılmamalı
- Etik sınırlar net söylenmeli
- Projenin çalışan demo olduğu vurgulanmalı

Kaçınılması gereken ifadeler:

- "Tanı koyuyoruz"
- "Hastalığı buluyor"
- "Tedavi öneriyor"
- "Klinik karar veriyor"
- "Gerçek hastalarda kullanılabilir"

Kullanılması önerilen ifadeler:

- "Klinik sinyal haritası"
- "Metinsel örüntü analizi"
- "Eğitim amaçlı prototip"
- "Sıralama payı"
- "Tıbbi karar değildir"
- "Etik sınırları belirlenmiş sistem"

## 8. Tasarim Onerileri

Görsel stil:

- Temiz ve modern
- Tıbbi/akademik his veren koyu ve ciddi bir tasarım
- Genel arka plan rengi `#61000a`
- Genel yazı rengi beyaz
- Vurgu için açık gri, beyaz veya çok açık pembe tonları
- Aşırı animasyon veya karmaşık grafik kullanılmamalı

Slayt başına önerilen metin miktarı:

- En fazla 4-5 kısa madde
- Teknik ayrıntılar konuşmacı notuna bırakılmalı
- Metrikler kart veya küçük tablo şeklinde gösterilmeli

Kullanılabilecek görsel öğeler:

- Veri akış diyagramı
- Modelleme pipeline şeması
- Metrik kartları
- Etik güvenlik kontrol listesi
- Demo ekranı ekran görüntüsü

## 9. Konusmaci Notlari - Kisa Versiyon

Sunum yaparken kullanılabilecek kısa anlatım:

### Açılış

> Bu projede House MD veri setinden yararlanarak klinik metinlerdeki semptom örüntülerini ve iletişim tonunu analiz eden bir NLP prototipi geliştirdik. Sistem tanı koymaz; eğitim amaçlı klinik sinyal haritası üretir.

### Veri

> Ham veri 7282 satır ve 16 sütundan oluşuyordu. Ana girdi olarak `text`, hedef olarak `Symptom` ve `Emotion` alanlarını kullandık. Veri kurgusal olduğu için gerçek hasta dağılımını temsil etmediğini özellikle belirttik.

### Ön İşleme

> Eksik verileri ve anlamsız etiketleri temizledik. Metinleri küçük harfe çevirdik, noktalama işaretlerini kaldırdık ve model için `clean_text` alanını oluşturduk.

### Modelleme

> İki model eğittik: klinik sinyal modeli ve iletişim tonu modeli. TF-IDF özellikleri ile LinearSVC kullandık. Bu bize hızlı, açıklanabilir ve ölçülebilir bir baseline sağladı.

### Sonuç

> Klinik sinyal modelinde accuracy 0.709, top-3 accuracy 0.848 elde ettik. İletişim tonu modelinde accuracy 0.542, top-3 accuracy 0.815 oldu.

### Etik

> Etik kısmı sadece rapora yazmadık; uygulamaya da ekledik. Sistem tanı dili kullanmıyor, analiz öncesi onay istiyor, acil durum ve kişisel veri uyarıları veriyor.

### Kapanış

> Sonuç olarak veri hazırlama, ön işleme, modelleme, canlıya alma, etik ve raporlama aşamalarını tamamlayan çalışan bir NLP prototipi geliştirdik.

## 10. Muhtemel Hoca Sorulari ve Cevaplar

### Bu sistem tani koyuyor mu?

Hayır. Sistem tanı koymaz. Sadece metindeki klinik sinyal örüntülerini sıralar. Bu yüzden arayüzde "tanı" yerine "klinik sinyal" dili kullanılmıştır.

### Neden en sık 50 semptom sınıfı seçildi?

Veri setinde 2562 benzersiz semptom etiketi vardı ve sınıflar çok dengesizdi. Birçok sınıfta çok az örnek bulunduğu için ölçülebilir ve dengeli bir baseline kurmak amacıyla en sık 50 sınıf seçildi.

### Neden BERT yerine TF-IDF + LinearSVC kullanıldı?

BERT proje için gelecek çalışma olarak planlandı. Bu aşamada hızlı eğitilebilen, açıklanabilir ve metrikleri kolay raporlanabilen bir baseline model tercih edildi.

### Etik kısmını nasıl yaptınız?

Etik uyarı sadece metin olarak bırakılmadı. Uygulamada etik onay kutusu, acil durum uyarısı, kişisel veri uyarısı, model kartı, risk kaydı ve veri beyanı eklendi.

### Veri gerçek hasta verisi mi?

Hayır. Veri House MD dizisi bağlamında hazırlanmış kurgusal bir veri setidir. Bu nedenle gerçek hasta dağılımını temsil etmez ve klinik kullanım iddiası taşımaz.

### Modelin en güçlü tarafı nedir?

Çalışan bir uçtan uca prototip olmasıdır: veri hazırlanmış, temizlenmiş, model eğitilmiş, metrikler çıkarılmış, canlı arayüz hazırlanmış ve etik güvenlik katmanı eklenmiştir.

### `ateş` ve `ates` model için aynı mı?

Tamamen aynı değildir. Model Türkçe veriyle eğitildiği için Türkçe karakterli girişler daha sağlıklıdır. Karakter TF-IDF sayesinde benzerlik kısmen yakalanır ama `ateş` ve `ates` farklı sonuçlar üretebilir.

### Top-3 Accuracy hangi yöntemle hesaplandı?

LinearSVC modelinin her sınıf için ürettiği karar skorları sıralandı. Gerçek etiket en yüksek skorlu ilk 3 sınıf içinde yer alıyorsa tahmin doğru kabul edildi.

### `class_weight="balanced"` nasıl çalışır?

Az örnekli sınıflara daha yüksek hata ağırlığı, çok örnekli sınıflara daha düşük hata ağırlığı verir. Böylece modelin sadece sık sınıflara yönelmesi azaltılmaya çalışılır.

## 11. Sunum Hazirlayacak AI Icin Net Prompt

Aşağıdaki prompt başka bir AI aracına verilebilir:

```text
Elindeki bilgilerle 12-13 slaytlık, anlaşılır ve akademik bir PowerPoint sunumu hazırla.

Proje adı: House MD Klinik Muhakeme AI Prototipi.

Ana mesaj: Bu proje gerçek tıbbi tanı koyan bir sistem değildir. House MD veri setindeki kurgusal klinik diyaloglardan klinik sinyal haritası ve iletişim tonu ipuçları çıkaran, etik güvenlik katmanı olan eğitim amaçlı bir NLP prototipidir.

Grup üyeleri:
- Ömer Faruk Pehlivan - 030123083
- Hüseyin Yusuf Cihangir - 030121037
- Faruk Laçin - 030122117

Görev dağılımı:
- Hüseyin Yusuf Cihangir: veri seti hazırlama, kolon kontrolü, veri ön işleme, temiz veri üretimi.
- Ömer Faruk Pehlivan: proje koordinasyonu, klinik sinyal modeli, modelleme stratejisi, Streamlit canlı arayüz, demo akışı.
- Faruk Laçin: iletişim tonu modeli, metrik raporlama, etik dokümantasyon, rapor ve sunum düzeni.

Sunum akışı:
1. Kapak
2. Projenin amacı
3. Ekip ve görev dağılımı
4. Veri seti
5. Ön işleme
6. Özellik seçimi
7. Modelleme
8. Model sonuçları
9. Canlı arayüz
10. Etik ve güvenlik katmanı
11. Demo senaryoları
12. Sonuç ve gelecek çalışmalar
13. Kısa teknik kavramlar / soru-cevap

Önemli sayılar:
- Ham veri: 7282 satır, 16 sütun
- İşlenmiş veri: 4068 satır, 18 sütun
- Benzersiz semptom etiketi: 2562
- Semptom modelinde kullanılan sınıf sayısı: 50
- Duygu/ton modelinde kullanılan sınıf sayısı: 10

Model:
- TF-IDF + LinearSVC
- %80 eğitim, %20 test
- class_weight="balanced"
- Stratified split

Semptom / klinik sinyal modeli:
- Kullanılan satır: 790
- Accuracy: 0.709
- Macro F1: 0.617
- Weighted F1: 0.684
- Top-3 Accuracy: 0.848

İletişim tonu modeli:
- Kullanılan satır: 3262
- Accuracy: 0.542
- Macro F1: 0.283
- Weighted F1: 0.520
- Top-3 Accuracy: 0.815

Canlı arayüz:
- Streamlit
- Yerel adres: http://localhost:8501
- Etik onay kutusu var
- Acil durum uyarısı var
- Kişisel veri uyarısı var

Etik mesaj:
- Sistem tanı koymaz.
- Tedavi önermez.
- Acil karar için kullanılamaz.
- Gerçek hasta verisi girilmemelidir.
- Veri kurgusaldır ve gerçek hasta dağılımını temsil etmez.

Kısa teknik açıklamalar:
- Benzersiz semptom etiketi, Symptom sütunundaki farklı sınıf sayısıdır; bizde 2562'dir.
- Nadir semptomlar ready_data.csv dosyasından silinmedi; yalnızca baseline eğitiminde en sık 50 semptom sınıfı kullanıldı.
- Duygu/iletişim tonu için en sık 10 sınıf kullanıldı çünkü veri dağılımı dengesizdi.
- Kelime TF-IDF, kelime ve iki kelimelik ifadeleri sayısal önem puanlarına dönüştürür.
- Karakter TF-IDF, kelimeleri 3-5 karakterlik parçalara ayırarak Türkçe ek ve yazım farklarını yakalar.
- LinearSVC, TF-IDF özelliklerini sınıflandıran hızlı doğrusal modeldir.
- Top-3 Accuracy, doğru etiketin modelin en yüksek skorlu ilk üç tahmini içinde bulunma oranıdır.
- class_weight="balanced", az örnekli sınıflara daha yüksek hata ağırlığı verir.
- Model Türkçe veriyle eğitilmiştir; ateş ve ates tamamen aynı değildir ama karakter TF-IDF benzerliği kısmen yakalar.
- Klinik Sinyal Haritası ve İletişim Tonu/Muhakeme İpuçları dili, tanı koyuyor algısını azaltmak için tercih edilmiştir.

Tasarım:
- Temiz, modern, akademik
- Genel arka plan rengi #61000a olsun
- Genel yazılar beyaz olsun
- Vurgu için açık gri, beyaz veya çok açık pembe kullanılabilir
- Slaytlarda kısa maddeler kullan
- Gereksiz kod detayına girme
- Metrikleri kart veya tablo olarak göster
- Etik slaytını güçlü ve net yap
```

## 12. Dosya ve Klasor Referanslari

Kod dosyaları:

```text
src/preprocess.py
src/train_baseline.py
src/predict_baseline.py
src/ethics.py
app.py
```

Veri dosyaları:

```text
data/raw/house_md_data.csv
data/processed/ready_data.csv
```

Model dosyaları:

```text
models/symptom_classifier.joblib
models/emotion_classifier.joblib
models/baseline_metadata.json
```

Rapor dosyaları:

```text
reports/PROJE_RAPORU.md
MODEL_CARD.md
reports/ETHICS_AND_GOVERNANCE.md
reports/RISK_REGISTER.md
reports/DATA_STATEMENT.md
```

Sunum dosyaları:

```text
reports/presentation/SUNUM_HAZIRLAMA_BRIEF.md
```
