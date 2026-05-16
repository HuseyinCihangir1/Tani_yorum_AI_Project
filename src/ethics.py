import re


EMERGENCY_TERMS = {
    "112",
    "acil",
    "kalp krizi",
    "inme",
    "felc",
    "felç",
    "bilinc kaybi",
    "bilinç kaybi",
    "bilincini kaybet",
    "bilincini yitir",
    "nefes alam",
    "solunum dur",
    "gogus agrisi",
    "göğüs ağrısı",
    "siddetli kanama",
    "şiddetli kanama",
    "kanama durmuyor",
    "zehirlen",
    "asiri doz",
    "aşırı doz",
    "intihar",
    "kendime zarar",
    "nobet",
    "nöbet",
}

PERSONAL_DATA_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?90\s*)?(?:0?\d{3}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2})(?!\d)"),
    "tc_like_id": re.compile(r"(?<!\d)\d{11}(?!\d)"),
}


def normalize_text(text):
    return str(text).casefold()


def detect_emergency_terms(text):
    normalized = normalize_text(text)
    return sorted(term for term in EMERGENCY_TERMS if term in normalized)


def detect_personal_data(text):
    findings = []
    for label, pattern in PERSONAL_DATA_PATTERNS.items():
        if pattern.search(str(text)):
            findings.append(label)
    return findings


def confidence_band(value):
    if value >= 0.45:
        return "yuksek siralama payi"
    if value >= 0.25:
        return "orta siralama payi"
    return "dusuk siralama payi"


def format_ethical_label(label):
    return str(label).replace("_", " ").strip()
