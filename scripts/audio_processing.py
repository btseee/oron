"""
Audio processing pipeline for Mongolian TTS dataset.
Includes: denoising, resampling, trimming, and normalization.
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


class AudioProcessor:
    def __init__(
        self,
        target_sr: int = 22050,
        trim_db: float = 20.0,
        trim_frame_length: int = 2048,
        trim_hop_length: int = 512,
        normalize_target_db: float = -20.0,
    ):
        self.target_sr = target_sr
        self.trim_db = trim_db
        self.trim_frame_length = trim_frame_length
        self.trim_hop_length = trim_hop_length
        self.normalize_target_db = normalize_target_db

    def load_audio(self, filepath: str) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=self.trim_db,
            frame_length=self.trim_frame_length,
            hop_length=self.trim_hop_length,
        )
        return trimmed

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (self.normalize_target_db / 20)
            audio = audio * (target_rms / rms)
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def process_single(self, input_path: str, output_path: str) -> bool:
        try:
            audio, sr = self.load_audio(input_path)
            audio = self.resample(audio, sr)
            audio = self.trim_silence(audio)
            audio = self.normalize(audio)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, self.target_sr)
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False


class DeepFilterNetDenoiser:
    def __init__(self, model_path: str = None):
        self.model_path = model_path

    def denoise_file(self, input_path: str, output_path: str) -> bool:
        try:
            cmd = ["deepFilter", input_path, "-o", output_path]
            if self.model_path:
                cmd.extend(["-m", self.model_path])
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"DeepFilterNet error for {input_path}: {e.stderr.decode()}")
            return False
        except FileNotFoundError:
            print("DeepFilterNet not found. Install with: pip install deepfilternet")
            return False

    def denoise_batch(
        self, 
        input_dir: str, 
        output_dir: str, 
        extension: str = ".wav"
    ) -> None:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            cmd = ["deepFilter", str(input_path), "-o", str(output_path)]
            if self.model_path:
                cmd.extend(["-m", self.model_path])
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Batch denoising failed: {e}")


class FFmpegConverter:
    @staticmethod
    def convert(
        input_path: str,
        output_path: str,
        sample_rate: int = 22050,
        channels: int = 1,
        bit_depth: int = 16,
    ) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-sample_fmt", f"s{bit_depth}",
                output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error for {input_path}: {e.stderr.decode()}")
            return False

    @staticmethod
    def mp3_to_wav(input_path: str, output_path: str, sample_rate: int = 22050) -> bool:
        return FFmpegConverter.convert(input_path, output_path, sample_rate)


def process_dataset(
    input_dir: str,
    output_dir: str,
    target_sr: int = 22050,
    denoise: bool = True,
    num_workers: int = 4,
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_path.glob("*.mp3")) + list(input_path.glob("*.wav"))
    
    stats = {"total": len(audio_files), "success": 0, "failed": 0}
    processor = AudioProcessor(target_sr=target_sr)
    
    temp_dir = output_path / "temp_wav"
    temp_dir.mkdir(exist_ok=True)
    
    print("Converting MP3 to WAV...")
    wav_files = []
    for audio_file in tqdm(audio_files, desc="Converting"):
        if audio_file.suffix == ".mp3":
            temp_wav = temp_dir / f"{audio_file.stem}.wav"
            if FFmpegConverter.mp3_to_wav(str(audio_file), str(temp_wav), target_sr):
                wav_files.append(temp_wav)
        else:
            wav_files.append(audio_file)

    if denoise:
        print("Denoising audio files...")
        denoised_dir = output_path / "denoised"
        denoiser = DeepFilterNetDenoiser()
        denoiser.denoise_batch(str(temp_dir), str(denoised_dir))
        wav_files = list(denoised_dir.glob("*.wav"))

    print("Processing audio files...")
    final_dir = output_path / "processed"
    final_dir.mkdir(exist_ok=True)

    def process_file(wav_file):
        output_file = final_dir / f"{wav_file.stem}.wav"
        return processor.process_single(str(wav_file), str(output_file))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in wav_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            if future.result():
                stats["success"] += 1
            else:
                stats["failed"] += 1

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio dataset for TTS")
    parser.add_argument("--input", required=True, help="Input directory with audio files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    stats = process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_sr=args.sample_rate,
        denoise=not args.no_denoise,
        num_workers=args.workers,
    )
    
    print(f"\nProcessing complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
