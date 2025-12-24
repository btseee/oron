import re
import unicodedata


MN_CYRILLIC = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
MN_CYRILLIC_UPPER = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
MN_PUNCTUATION = ".,!?"
MN_ALLOWED = MN_CYRILLIC + MN_PUNCTUATION + " "

UNITS = ["", "нэг", "хоёр", "гурав", "дөрөв", "тав", "зургаа", "долоо", "найм", "ес"]
TENS = ["", "арван", "хорин", "гучин", "дөчин", "тавин", "жаран", "далан", "наян", "ерэн"]
TENS_STANDALONE = ["", "арав", "хорь", "гуч", "дөч", "тавь", "жар", "дал", "ная", "ер"]

ABBREVIATIONS = {
    "т.ө": "түүнээс өмнө",
    "м.э.ө": "манай эриний өмнө",
    "м.э": "манай эрин",
    "жнь": "жишээ нь",
    "ж.нь": "жишээ нь",
    "г.м": "гэх мэт",
    "г.м.": "гэх мэт",
    "тэрб": "тэрбум",
    "мян": "мянга",
    "км": "километр",
    "см": "сантиметр",
    "мм": "миллиметр",
    "кг": "килограмм",
    "гр": "грамм",
    "мл": "миллилитр",
    "мин": "минут",
    "сек": "секунд",
    "ам.доллар": "америк доллар",
    "төг": "төгрөг",
    "₮": "төгрөг",
    "$": "доллар",
    "€": "евро",
    "№": "дугаар",
    "%": "хувь",
}

LATIN_TO_CYRILLIC = {
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф",
    "g": "г", "h": "х", "i": "и", "j": "ж", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "й", "z": "з",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def expand_number(num: int) -> str:
    if num == 0:
        return "тэг"
    if num < 0:
        return "хасах " + expand_number(-num)
    
    parts = []
    
    if num >= 1_000_000_000_000:
        trillions = num // 1_000_000_000_000
        num %= 1_000_000_000_000
        parts.append(expand_number(trillions) + " их наяд")
    
    if num >= 1_000_000_000:
        billions = num // 1_000_000_000
        num %= 1_000_000_000
        parts.append(expand_number(billions) + " тэрбум")
    
    if num >= 1_000_000:
        millions = num // 1_000_000
        num %= 1_000_000
        parts.append(expand_number(millions) + " сая")
    
    if num >= 1000:
        thousands = num // 1000
        num %= 1000
        if thousands == 1:
            parts.append("мянга")
        else:
            parts.append(expand_number(thousands) + " мянга")
    
    if num >= 100:
        hundreds = num // 100
        num %= 100
        if hundreds == 1:
            parts.append("зуу")
        else:
            parts.append(UNITS[hundreds] + " зуу")
    
    if num >= 10:
        tens = num // 10
        ones = num % 10
        if ones == 0:
            parts.append(TENS_STANDALONE[tens])
        else:
            parts.append(TENS[tens] + " " + UNITS[ones])
    elif num > 0:
        parts.append(UNITS[num])
    
    return " ".join(parts)


def expand_numbers(text: str) -> str:
    def replace_number(match):
        try:
            return expand_number(int(match.group(0)))
        except ValueError:
            return match.group(0)
    return re.sub(r"\d+", replace_number, text)


def expand_abbreviations(text: str) -> str:
    for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(re.escape(abbr), full, text, flags=re.IGNORECASE)
    return text


def remove_latin(text: str) -> str:
    result = []
    for char in text:
        lower = char.lower()
        if lower in LATIN_TO_CYRILLIC:
            result.append(LATIN_TO_CYRILLIC[lower])
        elif char.lower() not in "abcdefghijklmnopqrstuvwxyz":
            result.append(char)
    return "".join(result)


def normalize_punctuation(text: str) -> str:
    text = re.sub(r'[""„«»\'\"'']', '', text)
    text = re.sub(r'[–—−\-]+', ' ', text)
    text = re.sub(r'[;:\(\)\[\]\{\}]', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    return text


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.lower()
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = remove_latin(text)
    text = normalize_punctuation(text)
    text = re.sub(r'[^' + MN_CYRILLIC + MN_PUNCTUATION + r'\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def validate_text(text: str) -> bool:
    if not text:
        return False
    for char in text:
        if char not in MN_ALLOWED:
            return False
    return True


def has_latin(text: str) -> bool:
    return bool(re.search(r'[a-zA-Z]', text))


class MongolianTextCleaner:
    def __init__(self, min_length: int = 3, max_length: int = 300):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, text: str) -> str | None:
        text = clean_text(text)
        if not text or len(text) < self.min_length or len(text) > self.max_length:
            return None
        if not validate_text(text):
            return None
        return text
