"""
Custom data utilities for Mongolian TTS.
Wraps VITS2 data loaders with Mongolian text processing.
"""

import os
import sys
from pathlib import Path
from typing import List

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "vits2"))

from torchtext.vocab import vocab as build_vocab
from collections import OrderedDict

from src.mongolian.cleaners import (
    mongolian_clean_text,
    mongolian_add_spaces,
    mongolian_to_phonemes,
    mongolian_add_bos_eos,
    mongolian_delete_unks,
)
from src.mongolian.symbols import MONGOLIAN_SYMBOLS, UNK_ID


def load_mongolian_vocab(vocab_file: str):
    with open(vocab_file, "r", encoding="utf-8") as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    ordered_dict = OrderedDict((symbol, i) for i, symbol in enumerate(symbols))
    v = build_vocab(ordered_dict)
    v.set_default_index(UNK_ID)
    return v


def mongolian_tokenizer(text: str, vocab, cleaned_text: bool = False) -> List[int]:
    if cleaned_text:
        return list(map(int, text.split("\t")))
    
    text = mongolian_clean_text(text)
    text = mongolian_add_spaces(text)
    text = mongolian_to_phonemes(text)
    
    tokens = []
    for char in text:
        if char == " ":
            tokens.append(vocab["<space>"])
        else:
            idx = vocab[char] if char in vocab else UNK_ID
            tokens.append(idx)
    
    tokens = mongolian_add_bos_eos(tokens)
    tokens = mongolian_delete_unks(tokens)
    
    return tokens


def create_vocab_file(output_path: str):
    from src.mongolian.symbols import create_vocab_file as _create
    _create(output_path)
