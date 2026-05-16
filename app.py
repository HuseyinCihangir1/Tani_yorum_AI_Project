import streamlit as st

from src.ethics import (
    confidence_band,
    detect_emergency_terms,
    detect_personal_data,
    format_ethical_label,
)
from src.predict_baseline import load_model, predict_top_k


st.set_page_config(page_title="House MD Klinik Muhakeme Prototipi", layout="centered")


@st.cache_resource
def get_models():
    return {
        "symptom": load_model("symptom_classifier"),
        "emotion": load_model("emotion_classifier"),
    }


def render_signal_map(title, predictions, description):
    st.subheader(title)
    st.caption(description)

    for rank, item in enumerate(predictions, start=1):
        label = format_ethical_label(item["label"])
        share = item["confidence"]
        band = confidence_band(share)
        st.progress(share, text=f"{rank}. {label} | {band} | siralama payi: {share:.1%}")


def render_safety_findings(text):
    emergency_terms = detect_emergency_terms(text)
    personal_data = detect_personal_data(text)

    if emergency_terms:
        st.error(
            "Bu metin acil durum ifadesi icerebilir. Gercek bir acil durum varsa "
            "model sonucunu beklemeden 112 veya yerel acil saglik hizmetlerine basvurulmalidir."
        )
        st.caption(f"Yakalanan acil sinyal ifadeleri: {', '.join(emergency_terms[:8])}")

    if personal_data:
        st.warning(
            "Metin e-posta, telefon veya TC kimlik benzeri kisisel veri icerebilir. "
            "Proje demosunda gercek kisiye ait veri girilmemelidir."
        )
        st.caption(f"Yakalanan veri turleri: {', '.join(personal_data)}")


models = get_models()

st.title("House MD Klinik Muhakeme Prototipi")
st.caption("Kurgusal dizi verisi uzerinde egitim amacli NLP ve sorumlu yapay zeka demonstrasyonu")

with st.sidebar:
    st.header("Etik Cerceve")
    st.markdown(
        """
        - Tani koymaz, tedavi onermez.
        - Gercek hasta verisi girilmemelidir.
        - Ciktilar klinik karar yerine gecmez.
        - Acil durumda 112 / uzman saglik destegi onceliklidir.
        - Veri kaynagi kurgusaldir; gercek hasta dagilimini temsil etmez.
        """
    )

st.warning(
    "Bu sistem tibbi cihaz veya klinik karar sistemi degildir. Sadece House MD repliklerinden "
    "ogrenilen metin oruntulerini egitim amacli analiz eder."
)

sample = "Hasta ates, oksuruk ve nefes darligi yasiyor."
text = st.text_area("Kisisel veri icermeyen klinik metin", value=sample, height=150)
top_k = st.slider("Gosterilecek sinyal sayisi", min_value=3, max_value=10, value=5)

acknowledged = st.checkbox(
    "Bu prototipin egitim amacli oldugunu, tibbi tani/tedavi yerine kullanilamayacagini ve "
    "gercek hasta verisi girmemem gerektigini anladim."
)

submitted = st.button(
    "Etik Kosullari Kabul Ederek Analiz Et",
    type="primary",
    disabled=not acknowledged,
)

if not acknowledged:
    st.info("Analiz baslatmak icin etik kullanim onayini isaretleyin.")

if submitted:
    if not text.strip():
        st.error("Metin bos olamaz.")
    else:
        render_safety_findings(text)

        symptom_predictions = predict_top_k(models["symptom"], text, top_k=top_k)
        emotion_predictions = predict_top_k(models["emotion"], text, top_k=min(top_k, 5))

        render_signal_map(
            "Klinik Sinyal Haritasi",
            symptom_predictions,
            "Bu liste tani degil; metindeki ifadelerin egitim verisindeki semptom oruntuleriyle benzerlik siralamasidir.",
        )
        render_signal_map(
            "Iletisim Tonu ve Muhakeme Ipuclari",
            emotion_predictions,
            "Bu liste duygu/ton etiketlerinin metinle istatistiksel yakinligini gosterir; psikolojik degerlendirme degildir.",
        )

        with st.expander("Bu sonuclar nasil okunmali?"):
            st.markdown(
                """
                - `siralama payi`, gercek olasilik veya klinik guven skoru degildir.
                - Ilk siradaki sinyal bile kesin tani anlamina gelmez.
                - Model kurgusal dizi replikleriyle egitildigi icin gercek klinik kullanima uygun degildir.
                - Cikti, rapor ve sunum icin NLP prototipi demonstrasyonu olarak yorumlanmalidir.
                """
            )
