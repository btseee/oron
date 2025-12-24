import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
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
    min_speaker_minutes: float = 30.0
    train_ratio: float = 0.98
    val_ratio: float = 0.01


class BaseDatasetProcessor(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.cleaner = MongolianTextCleaner()

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_audio_path(self, row: pd.Series) -> str:
        pass

    def filter_by_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        if "gender" not in df.columns:
            return df
        return df[df["gender"].isin(GENDER_MAP.keys())]

    def add_speaker_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "client_id" in df.columns:
            client_ids = df["client_id"].unique()
            client_to_id = {cid: idx for idx, cid in enumerate(sorted(client_ids))}
            df["speaker_id"] = df["client_id"].map(client_to_id)
        else:
            df["speaker_id"] = 0
        
        if "gender" in df.columns:
            df["gender_id"] = df["gender"].map(GENDER_MAP)
        else:
            df["gender_id"] = 0
        return df

    def clean_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text_col = "sentence" if "sentence" in df.columns else "text"
        df["cleaned_sentence"] = df[text_col].apply(self.cleaner)
        df = df[df["cleaned_sentence"].notna()]
        return df

    def process(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.filter_quality(df)
        df = self.filter_by_gender(df)
        df = self.add_speaker_id(df)
        df = self.clean_sentences(df)
        return df


@dataclass
class CommonVoiceConfig(DatasetConfig):
    min_upvotes: int = 2
    max_downvotes: int = 0


class CommonVoiceProcessor(BaseDatasetProcessor):
    def __init__(self, config: CommonVoiceConfig):
        super().__init__(config)
        self.config: CommonVoiceConfig = config

    def load_data(self) -> pd.DataFrame:
        dfs = []
        for split in ["train", "dev", "test", "validated", "other"]:
            tsv_path = self.config.raw_dir / f"{split}.tsv"
            if tsv_path.exists():
                df = pd.read_csv(tsv_path, sep="\t")
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No TSV files found in {self.config.raw_dir}")
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["path"])
        return combined

    def load_clip_durations(self) -> pd.DataFrame:
        duration_path = self.config.raw_dir / "clip_durations.tsv"
        if duration_path.exists():
            return pd.read_csv(duration_path, sep="\t", header=None, names=["clip", "duration_ms"])
        return pd.DataFrame(columns=["clip", "duration_ms"])

    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["up_votes"] >= self.config.min_upvotes]
        df = df[df["down_votes"] <= self.config.max_downvotes]
        return df

    def filter_by_speaker_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        durations = self.load_clip_durations()
        if durations.empty:
            return df
        
        durations["clip"] = durations["clip"].apply(lambda x: Path(x).name)
        df = df.merge(durations, left_on="path", right_on="clip", how="left")
        df["duration_ms"] = df["duration_ms"].fillna(5000)
        
        speaker_time = df.groupby("client_id")["duration_ms"].sum() / 60000
        valid_speakers = speaker_time[speaker_time >= self.config.min_speaker_minutes].index
        
        return df[df["client_id"].isin(valid_speakers)]

    def get_audio_path(self, row: pd.Series) -> str:
        return row["path"]

    def process(self) -> pd.DataFrame:
        df = self.load_data()
        df = self.filter_quality(df)
        df = self.filter_by_gender(df)
        df = self.filter_by_speaker_duration(df)
        df = self.add_speaker_id(df)
        df = self.clean_sentences(df)
        return df


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
                text = row.get("cleaned_sentence", row.get("sentence", row.get("text", "")))
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


DATASET_PROCESSORS = {
    "common_voice": CommonVoiceProcessor,
}


def get_processor(dataset_type: str, config: DatasetConfig) -> BaseDatasetProcessor:
    if dataset_type not in DATASET_PROCESSORS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_PROCESSORS.keys())}")
    return DATASET_PROCESSORS[dataset_type](config)


def prepare_metadata(
    raw_dir: Path,
    output_dir: Path,
    clips_dir: Path,
    audio_dir: Path,
    dataset_type: str = "common_voice",
) -> pd.DataFrame:
    if dataset_type == "common_voice":
        config = CommonVoiceConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            clips_dir=clips_dir,
            min_upvotes=2,
            max_downvotes=0,
        )
    else:
        config = DatasetConfig(
            raw_dir=raw_dir,
            output_dir=output_dir,
            clips_dir=clips_dir,
        )
    
    processor = get_processor(dataset_type, config)
    df = processor.process()
    
    generator = FilelistGenerator(config, audio_dir)
    filelists = generator.generate_filelists(df)
    generator.save_filelists(filelists, output_dir / "filelists")
    
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    n_speakers = df["speaker_id"].nunique()
    speaker_info = {"n_speakers": n_speakers, "dataset_type": dataset_type}
    with open(output_dir / "speaker_info.json", "w") as f:
        json.dump(speaker_info, f)
    
    return df
