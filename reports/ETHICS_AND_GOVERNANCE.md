# Etik ve Yonetisim Plani

## Proje Konumlandirmasi

Bu proje bir tibbi cihaz, klinik karar destek sistemi veya tani sistemi olarak konumlandirilmamistir. Proje, kurgusal House MD replikleri uzerinde metin siniflandirma ve sorumlu yapay zeka demonstrasyonu amaciyla gelistirilmistir.

Ana etik ilke:

> Modelin ciktisi bilgi veya egitim demonstrasyonudur; tibbi tani, tedavi, acil karar veya hasta yonetimi yerine gecmez.

## Etik Tasarim Kararlari

1. Tani dili yerine sinyal dili kullanildi.

Uygulamada "Semptom / Tani Adaylari" basligi yerine "Klinik Sinyal Haritasi" kullanildi. Boylece modelin kesin tani verdigi izlenimi azaltildi.

2. Duygu analizi klinik psikolojik degerlendirme gibi sunulmadi.

"Duygu Analizi" yerine "Iletisim Tonu ve Muhakeme Ipuclari" basligi kullanildi. Bu cikti psikolojik tani degil, metinsel ton sinyali olarak aciklandi.

3. Kullanici onayi zorunlu hale getirildi.

Analiz butonu, kullanici egitim amacini ve tibbi karar yerine kullanilmamasi gerektigini kabul etmeden aktif olmaz.

4. Acil durum uyarisi eklendi.

Metinde "112", "kalp krizi", "nefes alam", "siddetli kanama", "intihar" gibi ifadeler yakalanirsa uygulama acil saglik destegi uyarisi verir.

5. Kisisel veri uyarisi eklendi.

E-posta, telefon ve TC kimlik benzeri veri desenleri yakalanirsa uygulama kullaniciyi uyarir.

6. Olasilik dili sinirlandi.

Model ciktisi "siralama payi" olarak gosterilir. Bu deger klinik guven, tani olasiligi veya risk skoru degildir.

## Veri Etigi

Veri kurgusal bir dizi kaynagindan geldigi icin gercek hasta verisi degildir. Bu durum gizlilik riskini azaltir ancak modelin klinik genellenebilirligini de sinirlar.

Gercek hasta verisi kullanilacaksa:

- Veriler anonimlestirilmeli veya uygun sekilde maskelenmelidir.
- Acik riza, hukuki dayanak ve kurum izni degerlendirilmelidir.
- Saglik verilerinin ozel nitelikli kisisel veri oldugu dikkate alinmalidir.
- Veri minimizasyonu uygulanmalidir.
- Model ciktisina erisim yetkilendirilmelidir.

## Canliya Alma Kontrol Listesi

- [x] Uygulamada tibbi kullanim siniri aciklandi.
- [x] Kullanici etik kullanim onayi eklendi.
- [x] Acil durum uyarisi eklendi.
- [x] Kisisel veri uyarisi eklendi.
- [x] Cikti dili tani iddiasindan uzaklastirildi.
- [x] Model karti hazirlandi.
- [x] Risk kaydi hazirlandi.
- [x] Veri beyan dosyasi hazirlandi.
- [ ] Klinik uzman tarafindan degerlendirme yapilmadi.
- [ ] Gercek hasta verisiyle validasyon yapilmadi.
- [ ] Model olasilik kalibrasyonu yapilmadi.

## Sunumda Kullanilacak Etik Mesaj

Bu proje yapay zekanin tibbi metinleri nasil isleyebilecegini gosteren bir egitim prototipidir. Modelin en onemli siniri, kurgusal veriyle egitilmis olmasi ve klinik validasyondan gecmemesidir. Bu nedenle sistem tani koymaz; yalnizca metindeki klinik sinyal ve iletisim tonu oruntulerini gosterir.

## Referanslar

- WHO safe and ethical AI for health: https://www.who.int/news/item/16-05-2023-who-calls-for-safe-and-ethical-ai-for-health
- European Commission, AI in healthcare: https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en
- KVKK ozel nitelikli kisisel veriler: https://www.kvkk.gov.tr/Icerik/2051/Ozel-Nitelikli-Kisisel-Veriler
