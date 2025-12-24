import re
import unicodedata


MN_CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
MN_PUNCTUATION = ".,!?;:—–-…\"'«»„""''"
MN_NUMBERS = "0123456789"
MN_ALLOWED = MN_CYRILLIC + MN_PUNCTUATION + MN_NUMBERS + " "

MN_NUMBER_WORDS = {
    "0": "тэг",
    "1": "нэг",
    "2": "хоёр",
    "3": "гурав",
    "4": "дөрөв",
    "5": "тав",
    "6": "зургаа",
    "7": "долоо",
    "8": "найм",
    "9": "ес",
    "10": "арав",
    "11": "арван нэг",
    "12": "арван хоёр",
    "13": "арван гурав",
    "14": "арван дөрөв",
    "15": "арван тав",
    "16": "арван зургаа",
    "17": "арван долоо",
    "18": "арван найм",
    "19": "арван ес",
    "20": "хорь",
    "30": "гуч",
    "40": "дөч",
    "50": "тавь",
    "60": "жар",
    "70": "дал",
    "80": "ная",
    "90": "ер",
    "100": "зуу",
    "1000": "мянга",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_punctuation(text: str) -> str:
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"\.{2,}", "…", text)
    return text


def expand_number(num: int) -> str:
    if num == 0:
        return MN_NUMBER_WORDS["0"]
    
    if num < 0:
        return "хасах " + expand_number(-num)
    
    if num <= 20:
        return MN_NUMBER_WORDS.get(str(num), str(num))
    
    if num < 100:
        tens = num // 10 * 10
        ones = num % 10
        result = MN_NUMBER_WORDS.get(str(tens), str(tens))
        if ones:
            result += " " + MN_NUMBER_WORDS[str(ones)]
        return result
    
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = MN_NUMBER_WORDS[str(hundreds)] + " зуу" if hundreds > 1 else "зуу"
        if remainder:
            result += " " + expand_number(remainder)
        return result
    
    if num < 10000:
        thousands = num // 1000
        remainder = num % 1000
        result = MN_NUMBER_WORDS[str(thousands)] + " мянга" if thousands > 1 else "мянга"
        if remainder:
            result += " " + expand_number(remainder)
        return result
    
    return str(num)


def expand_numbers(text: str) -> str:
    def replace_number(match):
        num_str = match.group(0)
        try:
            return expand_number(int(num_str))
        except ValueError:
            return num_str
    
    return re.sub(r"\d+", replace_number, text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = expand_numbers(text)
    text = re.sub(r"[^\w\s" + re.escape(MN_PUNCTUATION) + "]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def validate_text(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False
    for char in cleaned:
        if char not in MN_ALLOWED:
            return False
    return True


class MongolianTextCleaner:
    def __init__(self):
        self.min_length = 3
        self.max_length = 300

    def __call__(self, text: str) -> str | None:
        text = clean_text(text)
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return None
        if not validate_text(text):
            return None
        return text
