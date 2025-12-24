"""
Complete data preparation pipeline for Mongolian TTS.
Orchestrates audio processing and metadata preparation.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_processing import process_dataset
from scripts.metadata_processing import MetadataProcessor, create_vocab_file


def prepare_dataset(
    data_dir: str,
    output_dir: str,
    sample_rate: int = 22050,
    denoise: bool = True,
    num_workers: int = 4,
    min_upvotes: int = 2,
    max_downvotes: int = 1,
) -> dict:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clips_input = data_path / "clips"
    clips_output = output_path / "clips"

    print("=" * 60)
    print("STEP 1: Audio Processing")
    print("=" * 60)
    
    audio_stats = process_dataset(
        input_dir=str(clips_input),
        output_dir=str(clips_output),
        target_sr=sample_rate,
        denoise=denoise,
        num_workers=num_workers,
    )
    
    print(f"\nAudio processing complete:")
    print(f"  Processed: {audio_stats['success']}/{audio_stats['total']}")

    print("\n" + "=" * 60)
    print("STEP 2: Metadata Processing")
    print("=" * 60)

    filelists_dir = output_path / "filelists"
    processed_clips = clips_output / "processed"
    
    processor = MetadataProcessor(
        clips_dir=str(processed_clips),
        output_dir=str(filelists_dir),
        min_upvotes=min_upvotes,
        max_downvotes=max_downvotes,
    )

    metadata_stats = processor.process(
        train_tsv=str(data_path / "train.tsv"),
        dev_tsv=str(data_path / "dev.tsv") if (data_path / "dev.tsv").exists() else None,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Creating Vocab File")
    print("=" * 60)
    
    vocab_path = filelists_dir / "vocab.txt"
    create_vocab_file(str(vocab_path))

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print(f"Filelists: {filelists_dir}")
    print(f"Processed clips: {processed_clips}")
    print(f"\nDataset stats:")
    print(f"  Samples: {metadata_stats['n_samples']}")
    print(f"  Speakers: {metadata_stats['n_speakers']}")
    print(f"  Train/Val/Test: {metadata_stats['train_samples']}/{metadata_stats['val_samples']}/{metadata_stats['test_samples']}")

    return {
        "audio": audio_stats,
        "metadata": metadata_stats,
        "paths": {
            "clips": str(processed_clips),
            "filelists": str(filelists_dir),
            "vocab": str(vocab_path),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mongolian TTS dataset")
    parser.add_argument(
        "--data-dir",
        default="data/processed/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn",
        help="Common Voice data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared",
        help="Output directory",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--min-upvotes", type=int, default=2, help="Minimum upvotes filter")
    parser.add_argument("--max-downvotes", type=int, default=1, help="Maximum downvotes filter")

    args = parser.parse_args()

    prepare_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        denoise=not args.no_denoise,
        num_workers=args.workers,
        min_upvotes=args.min_upvotes,
        max_downvotes=args.max_downvotes,
    )
