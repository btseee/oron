import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from scripts.text.cleaner import MongolianTextCleaner


GENDER_MAP = {
    "male_masculine": 0,
    "female_feminine": 1,
}


@dataclass
class DatasetConfig:
    raw_dir: Path
    output_dir: Path
    clips_dir: Path
    min_votes: int = 2
    max_down_votes: int = 1
    train_ratio: float = 0.9
    val_ratio: float = 0.05


class MetadataProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cleaner = MongolianTextCleaner()

    def load_tsv(self, split: str) -> pd.DataFrame:
        tsv_path = self.config.raw_dir / f"{split}.tsv"
        return pd.read_csv(tsv_path, sep="\t")

    def filter_by_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["gender"].isin(GENDER_MAP.keys())]

    def filter_by_votes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["up_votes"] >= self.config.min_votes]
        df = df[df["down_votes"] <= self.config.max_down_votes]
        return df

    def add_speaker_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["speaker_id"] = df["gender"].map(GENDER_MAP)
        return df

    def clean_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cleaned_sentence"] = df["sentence"].apply(self.cleaner)
        df = df[df["cleaned_sentence"].notna()]
        return df

    def process(self) -> pd.DataFrame:
        dfs = []
        for split in ["train", "dev", "test"]:
            try:
                df = self.load_tsv(split)
                dfs.append(df)
            except FileNotFoundError:
                continue
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["path"])
        combined = self.filter_by_gender(combined)
        combined = self.filter_by_votes(combined)
        combined = self.add_speaker_id(combined)
        combined = self.clean_sentences(combined)
        
        return combined[["path", "sentence", "cleaned_sentence", "speaker_id", "gender"]]


class FilelistGenerator:
    def __init__(self, config: DatasetConfig, audio_dir: Path):
        self.config = config
        self.audio_dir = audio_dir

    def generate_filelists(self, df: pd.DataFrame) -> dict[str, list[str]]:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        splits = {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:],
        }
        
        filelists = {}
        for split_name, split_df in splits.items():
            lines = []
            for _, row in split_df.iterrows():
                wav_name = Path(row["path"]).stem + ".wav"
                wav_path = self.audio_dir / wav_name
                text = row.get("cleaned_sentence", row["sentence"])
                line = f"{wav_path}|{row['speaker_id']}|{text}"
                lines.append(line)
            filelists[split_name] = lines
        
        return filelists

    def save_filelists(self, filelists: dict[str, list[str]], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, lines in filelists.items():
            filepath = output_dir / f"{split_name}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


def prepare_metadata(
    raw_dir: Path,
    output_dir: Path,
    clips_dir: Path,
    audio_dir: Path,
) -> pd.DataFrame:
    config = DatasetConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        clips_dir=clips_dir,
    )
    
    processor = MetadataProcessor(config)
    df = processor.process()
    
    generator = FilelistGenerator(config, audio_dir)
    filelists = generator.generate_filelists(df)
    generator.save_filelists(filelists, output_dir / "filelists")
    
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    return df
