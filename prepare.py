#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from scripts.audio.processor import process_batch
from scripts.text.cleaner import MongolianTextCleaner
from scripts.metadata import prepare_metadata
from scripts.vocab import generate_vocab

load_dotenv()


def get_paths():
    return {
        "raw_dir": Path(os.getenv("DATA_RAW_DIR", "data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn")),
        "clips_dir": Path(os.getenv("CLIPS_DIR", "data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn/clips")),
        "output_dir": Path("data/prepared"),
        "audio_dir": Path("data/prepared/wavs"),
    }


def prepare_audio(paths: dict, denoise: bool = False):
    print("=== Processing Audio Files ===")
    
    clips_dir = paths["clips_dir"]
    audio_dir = paths["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    mp3_files = list(clips_dir.glob("*.mp3"))
    print(f"Found {len(mp3_files)} audio files")
    
    file_pairs = [
        (mp3_path, audio_dir / (mp3_path.stem + ".wav"))
        for mp3_path in mp3_files
    ]
    
    existing = [(i, o) for i, o in file_pairs if not o.exists()]
    print(f"Processing {len(existing)} new files (skipping {len(file_pairs) - len(existing)} existing)")
    
    if existing:
        results = process_batch(existing, denoise=denoise)
        print(f"Success: {results['success']}, Failed: {results['failed']}")


def prepare_text_and_metadata(paths: dict):
    print("\n=== Preparing Metadata ===")
    
    cleaner = MongolianTextCleaner()
    
    df = prepare_metadata(
        raw_dir=paths["raw_dir"],
        output_dir=paths["output_dir"],
        clips_dir=paths["clips_dir"],
        audio_dir=paths["audio_dir"],
    )
    
    print(f"Total samples: {len(df)}")
    print(f"Male samples: {len(df[df['speaker_id'] == 0])}")
    print(f"Female samples: {len(df[df['speaker_id'] == 1])}")


def prepare_vocab(paths: dict):
    print("\n=== Generating Vocabulary ===")
    
    vocab = generate_vocab(
        filelists_dir=paths["output_dir"] / "filelists",
        output_path=paths["output_dir"] / "vocab.txt",
    )
    
    print(f"Vocabulary size: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for VITS2 training")
    parser.add_argument("--denoise", action="store_true", help="Apply DeepFilterNet denoising")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio processing")
    args = parser.parse_args()
    
    paths = get_paths()
    
    if not args.skip_audio:
        prepare_audio(paths, denoise=args.denoise)
    
    prepare_text_and_metadata(paths)
    prepare_vocab(paths)
    
    print("\n=== Dataset Preparation Complete ===")
    print(f"Output directory: {paths['output_dir']}")


if __name__ == "__main__":
    main()
