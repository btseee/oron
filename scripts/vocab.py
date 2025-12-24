from pathlib import Path
from collections import Counter


MN_CYRILLIC_CHARS = list("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя")
PUNCTUATION = list(".,!?")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<space>"]


def build_vocab_from_texts(texts: list[str]) -> dict[str, int]:
    char_counter = Counter()
    
    for text in texts:
        for char in text:
            if char == " ":
                continue
            char_counter[char] += 1
    
    vocab = {}
    idx = 0
    
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1
    
    for char in MN_CYRILLIC_CHARS:
        if char not in vocab:
            vocab[char] = idx
            idx += 1
    
    for char in PUNCTUATION:
        if char not in vocab:
            vocab[char] = idx
            idx += 1
    
    for char, _ in char_counter.most_common():
        if char not in vocab and char.isprintable():
            vocab[char] = idx
            idx += 1
    
    return vocab


def save_vocab(vocab: dict[str, int], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\t{idx}\n")


def load_vocab(vocab_path: Path) -> dict[str, int]:
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


def text_to_tokens(text: str, vocab: dict[str, int]) -> list[int]:
    tokens = [vocab.get("<bos>", 2)]
    unk_id = vocab.get("<unk>", 1)
    space_id = vocab.get("<space>", 4)
    
    for char in text:
        if char == " ":
            tokens.append(space_id)
        else:
            tokens.append(vocab.get(char, unk_id))
    
    tokens.append(vocab.get("<eos>", 3))
    return tokens


def tokens_to_text(tokens: list[int], vocab: dict[str, int]) -> str:
    idx_to_char = {v: k for k, v in vocab.items()}
    chars = []
    
    for token in tokens:
        char = idx_to_char.get(token, "")
        if char == "<space>":
            chars.append(" ")
        elif not char.startswith("<"):
            chars.append(char)
    
    return "".join(chars)


def generate_vocab(filelists_dir: Path, output_path: Path) -> dict[str, int]:
    texts = []
    
    for filepath in filelists_dir.glob("*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    texts.append(parts[2])
    
    vocab = build_vocab_from_texts(texts)
    save_vocab(vocab, output_path)
    
    return vocab
