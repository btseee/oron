"""
Mongolian text cleaners for VITS2 TTS.
"""

import re
from typing import List

from src.mongolian.symbols import PAD_ID, UNK_ID, BOS_ID, EOS_ID


_whitespace_re = re.compile(r"\s+")

MONGOLIAN_CYRILLIC = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
MONGOLIAN_PUNCTUATION = ".,!?;:—–-\"'()«»"
MONGOLIAN_NUMBER_WORDS = {
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
    "100": "зуу",
    "1000": "мянга",
}


def mongolian_lowercase(text: str, *args, **kwargs) -> str:
    return text.lower()


def mongolian_collapse_whitespace(text: str, *args, **kwargs) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def mongolian_normalize_numbers(text: str, *args, **kwargs) -> str:
    def convert_number(match):
        num = match.group()
        if num in MONGOLIAN_NUMBER_WORDS:
            return MONGOLIAN_NUMBER_WORDS[num]
        
        result = []
        num_str = str(int(num))
        length = len(num_str)
        
        for i, digit in enumerate(num_str):
            if digit == "0":
                continue
            pos = length - i - 1
            word = MONGOLIAN_NUMBER_WORDS.get(digit, digit)
            
            if pos == 3:
                result.append(word + " мянга")
            elif pos == 2:
                result.append(word + " зуу")
            elif pos == 1:
                if digit == "1":
                    result.append("арав")
                else:
                    result.append(word + "н")
            else:
                result.append(word)
        
        return " ".join(result) if result else MONGOLIAN_NUMBER_WORDS["0"]
    
    return re.sub(r"\d+", convert_number, text)


def mongolian_clean_text(text: str, *args, **kwargs) -> str:
    text = mongolian_lowercase(text)
    text = mongolian_normalize_numbers(text)
    text = re.sub(r"[^\u0400-\u04FF\s.,!?;:—–\-\"'()«»]", "", text)
    text = mongolian_collapse_whitespace(text)
    return text


def mongolian_add_spaces(text: str, *args, **kwargs) -> str:
    text = re.sub(r"([.,!?;:—–\-\"'()«»])", r" \1 ", text)
    text = mongolian_collapse_whitespace(text)
    return text


def mongolian_to_phonemes(text: str, *args, **kwargs) -> str:
    """Convert Mongolian text to phoneme-like representation."""
    long_vowels = {"аа": "а:", "ээ": "э:", "ии": "и:", "оо": "о:", "уу": "у:", "өө": "ө:", "үү": "ү:"}
    
    for lv, phoneme in long_vowels.items():
        text = text.replace(lv, phoneme)
    
    text = text.replace("ь", "ʲ")
    text = text.replace("ъ", "")
    
    return text


def mongolian_tokenize(text: str, vocab, *args, **kwargs) -> List[int]:
    tokens = list(text)
    return vocab(tokens)


def mongolian_add_bos_eos(tokens: List[int], *args, **kwargs) -> List[int]:
    return [BOS_ID] + tokens + [EOS_ID]


def mongolian_add_blank(tokens: List[int], *args, **kwargs) -> List[int]:
    result = [PAD_ID] * (len(tokens) * 2 + 1)
    result[1::2] = tokens
    return result


def mongolian_delete_unks(tokens: List[int], *args, **kwargs) -> List[int]:
    return [token for token in tokens if token != UNK_ID]
