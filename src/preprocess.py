import pandas as pd
import re
import os

def preprocess_data():
    input_path = 'data/raw/house_md_data.csv' # [cite: 25, 104]
    output_path = 'data/processed/ready_data.csv' # [cite: 77]
    
    # 1. Dosya Varlık Kontrolü [cite: 86, 87, 102]
    if not os.path.exists(input_path):
        print(f"HATA: '{input_path}' dosyası bulunamadı. Lütfen dosya ismini ve konumunu kontrol edin.")
        return

    print(f"Dosya okunuyor: {input_path}")
    
    try:
        # 2. Karakter Kodlaması ve Tokenization Hatalarına Karşı Güvenli Okuma [cite: 91, 101]
        # 'sep' ve 'on_bad_lines' parametreleri tokenization hatalarını (sütun kayması vb.) engeller
        df = pd.read_csv(
            input_path, 
            encoding='ISO-8859-9', # Türkçe karakterler için [cite: 92, 101]
            sep=None,              # Ayırıcıyı (virgül/noktalı virgül) otomatik algılar
            engine='python',       # Otomatik ayırıcı algılama için gereklidir
            on_bad_lines='warn'    # Hatalı satırları atlar ve uyarır
        )
        
        # 3. Temizleme ve Ön İşleme [cite: 27, 77]
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text) # Noktalama işaretlerini kaldırır
            return text

        if 'text' in df.columns:
            df['clean_text'] = df['text'].apply(clean_text)
            # Hedef değişkeni (Label) belirleme [cite: 28, 39]
            if 'Symptom' in df.columns:
                df['label'] = pd.Categorical(df['Symptom']).codes
            
            # 4. UTF-8 Olarak Kaydetme (BERT uyumu için) [cite: 95]
            os.makedirs('data/processed', exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"BAŞARILI: Veri işlendi ve '{output_path}' konumuna kaydedildi.")
        else:
            print("HATA: Veri setinde 'text' sütunu bulunamadı!")

    except Exception as e:
        print(f"BEKLENMEYEN HATA OLUŞTU: {e}")

if __name__ == "__main__":
    preprocess_data()