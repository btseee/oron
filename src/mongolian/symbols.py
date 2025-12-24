"""Mongolian Cyrillic symbol set for VITS2 TTS."""

MONGOLIAN_CYRILLIC_LOWER = list("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя")
MONGOLIAN_PUNCTUATION = list(".,!?;:—–-\"'()«»")
MONGOLIAN_SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<space>"]
MONGOLIAN_PHONEME_EXTRAS = [":", "ʲ"]

MONGOLIAN_SYMBOLS = (
    MONGOLIAN_SPECIAL + 
    MONGOLIAN_CYRILLIC_LOWER + 
    MONGOLIAN_PUNCTUATION +
    MONGOLIAN_PHONEME_EXTRAS
)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPACE_ID = 4


def create_vocab_file(filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for symbol in MONGOLIAN_SYMBOLS:
            f.write(f"{symbol}\n")


if __name__ == "__main__":
    print(f"Total symbols: {len(MONGOLIAN_SYMBOLS)}")
    print(f"Symbols: {MONGOLIAN_SYMBOLS}")
