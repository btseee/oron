#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src" / "vits2"))

from model.models import SynthesizerTrn
from utils.hparams import get_hparams_from_file
from scripts.vocab import load_vocab, text_to_tokens
from scripts.text.cleaner import clean_text

load_dotenv()


def load_model(checkpoint_path: Path, config_path: Path, device: str = "cuda"):
    hps = get_hparams_from_file(str(config_path))
    vocab = load_vocab(Path(hps.data.vocab_file))
    
    model = SynthesizerTrn(
        len(vocab),
        hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, hps, vocab


def synthesize(
    text: str,
    model,
    hps,
    vocab: dict,
    speaker_id: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    text = clean_text(text)
    tokens = text_to_tokens(text, vocab)
    
    x = torch.LongTensor(tokens).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(tokens)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device) if hps.data.n_speakers > 1 else None
    
    with torch.no_grad():
        audio = model.infer(x, x_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0, 0]
    
    return audio.cpu()


def main():
    parser = argparse.ArgumentParser(description="VITS2 Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (0=male, 1=female)")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    args = parser.parse_args()
    
    model_dir = Path(os.getenv("MODEL_DIR", "checkpoints/oron"))
    config_path = Path("configs/config.yaml")
    
    checkpoint_path = args.checkpoint or sorted(model_dir.glob("G_*.pth"))[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, hps, vocab = load_model(checkpoint_path, config_path, device)
    audio = synthesize(args.text, model, hps, vocab, args.speaker, device)
    
    import torchaudio
    torchaudio.save(args.output, audio.unsqueeze(0), hps.data.sample_rate)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
