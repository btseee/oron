import subprocess
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class AudioProcessor:
    def __init__(
        self,
        target_sr: int = 22050,
        trim_db: float = 20.0,
        target_peak: float = 0.95,
    ):
        self.target_sr = target_sr
        self.trim_db = trim_db
        self.target_peak = target_peak

    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        audio, _ = librosa.effects.trim(audio, top_db=self.trim_db)
        return audio

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (self.target_peak / peak)
        return audio

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        try:
            audio, sr = librosa.load(input_path, sr=None, mono=True)
            audio = self.resample(audio, sr)
            audio = self.trim_silence(audio)
            audio = self.normalize(audio)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False


class DeepFilterProcessor:
    def __init__(self, post_filter: bool = True):
        self.post_filter = post_filter

    def denoise_file(self, input_path: Path, output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "deepfilter",
                str(input_path),
                "-o", str(output_path.parent),
                "-D", str(output_path.parent),
            ]
            if self.post_filter:
                cmd.append("-p")
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception as e:
            print(f"Error denoising {input_path}: {e}")
            return False


def process_audio_pipeline(
    input_path: Path,
    output_path: Path,
    denoise: bool = True,
    target_sr: int = 22050,
) -> bool:
    processor = AudioProcessor(target_sr=target_sr)
    
    if denoise:
        denoiser = DeepFilterProcessor()
        temp_path = output_path.with_suffix('.temp.wav')
        if not denoiser.denoise_file(input_path, temp_path):
            return processor.process_file(input_path, output_path)
        result = processor.process_file(temp_path, output_path)
        temp_path.unlink(missing_ok=True)
        return result
    
    return processor.process_file(input_path, output_path)


def process_batch(
    file_pairs: list[tuple[Path, Path]],
    denoise: bool = True,
    target_sr: int = 22050,
    max_workers: int = 8,
) -> dict:
    results = {"success": 0, "failed": 0}
    
    def process_single(pair):
        input_path, output_path = pair
        return process_audio_pipeline(input_path, output_path, denoise, target_sr)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for success in tqdm(executor.map(process_single, file_pairs), total=len(file_pairs)):
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
    
    return results
