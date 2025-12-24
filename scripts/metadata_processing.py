"""
Metadata processing for Common Voice Mongolian dataset.
Filters by gender and prepares VITS2 filelists.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mongolian.cleaners import mongolian_clean_text


GENDER_MAP = {
    "male_masculine": 0,
    "female_feminine": 1,
}


class MetadataProcessor:
    def __init__(
        self,
        clips_dir: str,
        output_dir: str,
        min_upvotes: int = 2,
        max_downvotes: int = 1,
    ):
        self.clips_dir = Path(clips_dir)
        self.output_dir = Path(output_dir)
        self.min_upvotes = min_upvotes
        self.max_downvotes = max_downvotes
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_tsv(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, sep="\t", dtype=str)

    def filter_by_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_genders = list(GENDER_MAP.keys())
        return df[df["gender"].isin(valid_genders)].copy()

    def filter_by_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up_votes"] = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0)
        df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0)
        
        mask = (df["up_votes"] >= self.min_upvotes) & (df["down_votes"] <= self.max_downvotes)
        return df[mask].copy()

    def filter_existing_files(
        self, 
        df: pd.DataFrame, 
        extension: str = ".wav"
    ) -> pd.DataFrame:
        existing = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
            filename = row["path"].replace(".mp3", extension)
            filepath = self.clips_dir / filename
            if filepath.exists():
                existing.append(True)
            else:
                existing.append(False)
        df = df[existing].copy()
        return df

    def clean_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df["cleaned_sentence"] = df["sentence"].apply(mongolian_clean_text)
        df = df[df["cleaned_sentence"].str.len() > 0].copy()
        return df

    def create_speaker_mapping(self, df: pd.DataFrame) -> dict:
        unique_clients = df["client_id"].unique()
        return {client: idx for idx, client in enumerate(unique_clients)}

    def create_filelist(
        self,
        df: pd.DataFrame,
        speaker_map: dict,
        output_name: str,
        use_cleaned: bool = True,
    ) -> str:
        output_path = self.output_dir / output_name
        
        lines = []
        for _, row in df.iterrows():
            audio_path = str(self.clips_dir / row["path"].replace(".mp3", ".wav"))
            speaker_id = speaker_map[row["client_id"]]
            text = row["cleaned_sentence"] if use_cleaned else row["sentence"]
            lines.append(f"{audio_path}|{speaker_id}|{text}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return str(output_path)

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        return train_df, val_df, test_df

    def process(
        self,
        train_tsv: str,
        dev_tsv: Optional[str] = None,
        test_tsv: Optional[str] = None,
    ) -> dict:
        print("Loading train data...")
        df = self.load_tsv(train_tsv)
        print(f"  Loaded {len(df)} samples")

        if dev_tsv:
            dev_df = self.load_tsv(dev_tsv)
            df = pd.concat([df, dev_df], ignore_index=True)
            print(f"  Added {len(dev_df)} dev samples")

        print("Filtering by gender...")
        df = self.filter_by_gender(df)
        print(f"  {len(df)} samples with valid gender")

        print("Filtering by quality...")
        df = self.filter_by_quality(df)
        print(f"  {len(df)} samples passed quality filter")

        print("Cleaning sentences...")
        df = self.clean_sentences(df)
        print(f"  {len(df)} samples after text cleaning")

        print("Checking existing audio files...")
        df = self.filter_existing_files(df)
        print(f"  {len(df)} samples with existing audio")

        print("Creating speaker mapping...")
        speaker_map = self.create_speaker_mapping(df)
        n_speakers = len(speaker_map)
        print(f"  {n_speakers} unique speakers")

        speaker_map_path = self.output_dir / "speaker_map.txt"
        with open(speaker_map_path, "w", encoding="utf-8") as f:
            for client_id, idx in speaker_map.items():
                gender = df[df["client_id"] == client_id]["gender"].iloc[0]
                f.write(f"{idx}|{gender}|{client_id[:16]}...\n")

        print("Splitting dataset...")
        train_df, val_df, test_df = self.split_dataset(df)
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        print("Creating filelists...")
        train_path = self.create_filelist(train_df, speaker_map, "train.txt")
        val_path = self.create_filelist(val_df, speaker_map, "val.txt")
        test_path = self.create_filelist(test_df, speaker_map, "test.txt")

        return {
            "n_samples": len(df),
            "n_speakers": n_speakers,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "filelists": {
                "train": train_path,
                "val": val_path,
                "test": test_path,
            },
            "gender_distribution": df["gender"].value_counts().to_dict(),
        }


def create_vocab_file(output_path: str) -> None:
    from src.mongolian.symbols import MONGOLIAN_SYMBOLS
    
    with open(output_path, "w", encoding="utf-8") as f:
        for symbol in MONGOLIAN_SYMBOLS:
            f.write(f"{symbol}\n")
    print(f"Created vocab file with {len(MONGOLIAN_SYMBOLS)} symbols at {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Common Voice metadata")
    parser.add_argument("--data-dir", required=True, help="Common Voice data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for filelists")
    parser.add_argument("--clips-dir", help="Processed clips directory (default: data-dir/clips)")
    parser.add_argument("--min-upvotes", type=int, default=2, help="Minimum upvotes")
    parser.add_argument("--max-downvotes", type=int, default=1, help="Maximum downvotes")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    clips_dir = args.clips_dir or str(data_dir / "clips")
    
    processor = MetadataProcessor(
        clips_dir=clips_dir,
        output_dir=args.output_dir,
        min_upvotes=args.min_upvotes,
        max_downvotes=args.max_downvotes,
    )
    
    stats = processor.process(
        train_tsv=str(data_dir / "train.tsv"),
        dev_tsv=str(data_dir / "dev.tsv"),
    )
    
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Speakers: {stats['n_speakers']}")
    print(f"  Gender distribution: {stats['gender_distribution']}")
    
    vocab_path = Path(args.output_dir) / "vocab.txt"
    create_vocab_file(str(vocab_path))
