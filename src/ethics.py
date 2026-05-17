import re
from dataclasses import dataclass


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(\+90|0)?\s?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}")
TC_ID_PATTERN = re.compile(r"\b[1-9]\d{10}\b")

EMERGENCY_PATTERNS = {
    "nefes alamama": re.compile(r"\b(nefes alamıyorum|nefes alamiyor|nefes alamıyor|boğuluyorum)\b", re.I),
    "göğüs ağrısı": re.compile(r"\b(göğüs ağrısı|gogus agrisi|kalp krizi)\b", re.I),
    "bilinç kaybı": re.compile(r"\b(bilinç kaybı|bilinc kaybi|bayıldı|bayildi|komaya girdi)\b", re.I),
    "felç belirtisi": re.compile(r"\b(felç|felc|yüz kayması|konuşma bozukluğu)\b", re.I),
    "şiddetli kanama": re.compile(r"\b(şiddetli kanama|siddetli kanama|durmayan kanama)\b", re.I),
    "intihar riski": re.compile(r"\b(intihar|kendime zarar|kendini öldür)\b", re.I),
    "zehirlenme": re.compile(r"\b(zehirlenme|zehir içti|zehir icti|aşırı doz|asiri doz)\b", re.I),
    "acil servis": re.compile(r"\b(112|acil servis|ambulans)\b", re.I),
}


@dataclass(frozen=True)
class EthicsReview:
    emergency_matches: list[str]
    personal_data_matches: list[str]

    @property
    def has_emergency_risk(self) -> bool:
        return bool(self.emergency_matches)

    @property
    def has_personal_data(self) -> bool:
        return bool(self.personal_data_matches)


def review_text(text: str) -> EthicsReview:
    emergency_matches = [
        label for label, pattern in EMERGENCY_PATTERNS.items() if pattern.search(text)
    ]

    personal_data_matches = []
    if EMAIL_PATTERN.search(text):
        personal_data_matches.append("e-posta")
    if PHONE_PATTERN.search(text):
        personal_data_matches.append("telefon")
    if TC_ID_PATTERN.search(text):
        personal_data_matches.append("TC kimlik benzeri numara")

    return EthicsReview(
        emergency_matches=emergency_matches,
        personal_data_matches=personal_data_matches,
    )
