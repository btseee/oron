#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from scripts.audio.processor import process_batch
from scripts.metadata import prepare_metadata, DATASET_PROCESSORS
from scripts.vocab import generate_vocab

load_dotenv()


def get_paths():
    return {
        "raw_dir": Path(os.getenv("DATA_RAW_DIR", "data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn")),
        "clips_dir": Path(os.getenv("CLIPS_DIR", "data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn/clips")),
        "output_dir": Path("data/prepared"),
        "audio_dir": Path("data/prepared/wavs"),
    }


def prepare_audio(paths: dict, denoise: bool = True, lufs_normalize: bool = True):
    print("=== Processing Audio Files ===")
    print(f"Denoise: {denoise}, LUFS Normalize: {lufs_normalize}")
    
    clips_dir = paths["clips_dir"]
    audio_dir = paths["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    file_pairs = [
        (f, audio_dir / (f.stem + ".wav"))
        for f in audio_files
    ]
    
    existing = [(i, o) for i, o in file_pairs if not o.exists()]
    print(f"Processing {len(existing)} new files (skipping {len(file_pairs) - len(existing)} existing)")
    
    if existing:
        results = process_batch(existing, denoise=denoise, lufs_normalize=lufs_normalize, max_workers=4)
        print(f"Success: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped (duration): {results['skipped_duration']}")


def prepare_text_and_metadata(paths: dict, dataset_type: str = "common_voice"):
    print(f"\n=== Preparing Metadata ({dataset_type}) ===")
    
    df = prepare_metadata(
        raw_dir=paths["raw_dir"],
        output_dir=paths["output_dir"],
        clips_dir=paths["clips_dir"],
        audio_dir=paths["audio_dir"],
        dataset_type=dataset_type,
    )
    
    n_speakers = df["speaker_id"].nunique()
    print(f"Total samples: {len(df)}")
    print(f"Total speakers: {n_speakers}")
    
    if "gender_id" in df.columns:
        print(f"Male samples: {len(df[df['gender_id'] == 0])}")
        print(f"Female samples: {len(df[df['gender_id'] == 1])}")


def prepare_vocab(paths: dict):
    print("\n=== Generating Vocabulary ===")
    
    vocab = generate_vocab(
        filelists_dir=paths["output_dir"] / "filelists",
        output_path=paths["output_dir"] / "vocab.txt",
    )
    
    print(f"Vocabulary size: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for VITS2 training")
    parser.add_argument("--dataset", type=str, default="common_voice", 
                        choices=list(DATASET_PROCESSORS.keys()),
                        help="Dataset type")
    parser.add_argument("--denoise", action="store_true", help="Apply DeepFilterNet3 denoising")
    parser.add_argument("--no-lufs", action="store_true", help="Skip LUFS normalization")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio processing")
    args = parser.parse_args()
    
    paths = get_paths()
    
    if not args.skip_audio:
        prepare_audio(paths, denoise=args.denoise, lufs_normalize=not args.no_lufs)
    
    prepare_text_and_metadata(paths, dataset_type=args.dataset)
    prepare_vocab(paths)
    
    print("\n=== Dataset Preparation Complete ===")
    print(f"Output directory: {paths['output_dir']}")


if __name__ == "__main__":
    main()
