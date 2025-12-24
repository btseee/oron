import subprocess
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


MIN_DURATION = 1.0
MAX_DURATION = 12.0
SILENCE_PADDING_MS = 75
TARGET_LUFS = -23.0
TARGET_PEAK_DB = -1.0


class AudioProcessor:
    def __init__(
        self,
        target_sr: int = 22050,
        trim_db: float = 30.0,
        min_duration: float = MIN_DURATION,
        max_duration: float = MAX_DURATION,
        silence_padding_ms: int = SILENCE_PADDING_MS,
    ):
        self.target_sr = target_sr
        self.trim_db = trim_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_padding_samples = int(silence_padding_ms * target_sr / 1000)

    def load_audio(self, input_path: Path) -> tuple[np.ndarray, int] | None:
        try:
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            return audio, sr
        except Exception:
            return None

    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        audio, _ = librosa.effects.trim(audio, top_db=self.trim_db)
        return audio

    def add_silence_padding(self, audio: np.ndarray) -> np.ndarray:
        padding = np.zeros(self.silence_padding_samples, dtype=audio.dtype)
        return np.concatenate([padding, audio, padding])

    def normalize_peak(self, audio: np.ndarray, target_db: float = TARGET_PEAK_DB) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            target_peak = 10 ** (target_db / 20)
            audio = audio * (target_peak / peak)
        return audio

    def check_duration(self, audio: np.ndarray) -> bool:
        duration = len(audio) / self.target_sr
        return self.min_duration <= duration <= self.max_duration

    def process_file(self, input_path: Path, output_path: Path) -> dict:
        result = {"success": False, "path": str(input_path), "reason": None}
        
        loaded = self.load_audio(input_path)
        if loaded is None:
            result["reason"] = "load_failed"
            return result
        
        audio, sr = loaded
        audio = self.resample(audio, sr)
        audio = self.trim_silence(audio)
        
        if not self.check_duration(audio):
            result["reason"] = "duration_invalid"
            return result
        
        audio = self.add_silence_padding(audio)
        audio = self.normalize_peak(audio)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
        
        result["success"] = True
        return result


class DeepFilterProcessor:
    def __init__(self, model: str = "DeepFilterNet3"):
        self.model = model

    def denoise_file(self, input_path: Path, output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "deepfilter",
                str(input_path),
                "-o", str(output_path.parent),
                "-D", str(output_path.parent),
                "-m", self.model,
                "-p",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            
            expected_output = output_path.parent / input_path.name
            if expected_output.exists() and expected_output != output_path:
                expected_output.rename(output_path)
            
            return output_path.exists()
        except Exception:
            return False


class LUFSNormalizer:
    def __init__(self, target_lufs: float = TARGET_LUFS):
        self.target_lufs = target_lufs

    def normalize_file(self, input_path: Path, output_path: Path) -> bool:
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-af", f"loudnorm=I={self.target_lufs}:TP=-1.5:LRA=11",
                "-ar", "22050", "-ac", "1",
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return output_path.exists()
        except Exception:
            return False


def process_audio_pipeline(
    input_path: Path,
    output_path: Path,
    denoise: bool = True,
    lufs_normalize: bool = True,
    target_sr: int = 22050,
) -> dict:
    processor = AudioProcessor(target_sr=target_sr)
    temp_dir = output_path.parent / ".temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    current_path = input_path
    
    if denoise:
        denoiser = DeepFilterProcessor()
        denoised_path = temp_dir / f"{input_path.stem}_denoised.wav"
        if denoiser.denoise_file(current_path, denoised_path):
            current_path = denoised_path
    
    if lufs_normalize:
        normalizer = LUFSNormalizer()
        normalized_path = temp_dir / f"{input_path.stem}_normalized.wav"
        if normalizer.normalize_file(current_path, normalized_path):
            current_path = normalized_path
    
    result = processor.process_file(current_path, output_path)
    
    for temp_file in temp_dir.glob(f"{input_path.stem}_*"):
        temp_file.unlink(missing_ok=True)
    
    return result


def process_single(args: tuple) -> dict:
    input_path, output_path, denoise, lufs_normalize, target_sr = args
    return process_audio_pipeline(input_path, output_path, denoise, lufs_normalize, target_sr)


def process_batch(
    file_pairs: list[tuple[Path, Path]],
    denoise: bool = True,
    lufs_normalize: bool = True,
    target_sr: int = 22050,
    max_workers: int = 4,
) -> dict:
    results = {"success": 0, "failed": 0, "skipped_duration": 0, "errors": []}
    
    args_list = [(i, o, denoise, lufs_normalize, target_sr) for i, o in file_pairs]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(process_single, args_list), total=len(file_pairs)):
            if result["success"]:
                results["success"] += 1
            elif result["reason"] == "duration_invalid":
                results["skipped_duration"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(result)
    
    return results
