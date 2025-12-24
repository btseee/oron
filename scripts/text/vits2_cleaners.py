import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "vits2"))

from scripts.text.cleaner import clean_text
from scripts.vocab import text_to_tokens, load_vocab


def mongolian_cleaners(text: str, vocab=None, *args, **kwargs) -> list[int]:
    text = clean_text(text)
    if vocab is None:
        vocab_path = Path("data/prepared/vocab.txt")
        vocab = load_vocab(vocab_path)
    return text_to_tokens(text, vocab)
