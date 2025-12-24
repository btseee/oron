"""
Inference script for Mongolian VITS2 model.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
VITS2_DIR = PROJECT_ROOT / "src" / "vits2"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VITS2_DIR))


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    from utils.hparams import HParams
    from model.models import SynthesizerTrn
    from src.mongolian.data_utils import load_mongolian_vocab

    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)
    
    hps = HParams(**config)
    vocab = load_mongolian_vocab(hps.data.vocab_file)

    net_g = SynthesizerTrn(
        len(vocab),
        hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_g.load_state_dict(checkpoint["model"])
    net_g.eval()

    return net_g, hps, vocab


def text_to_sequence(text: str, vocab):
    from src.mongolian.data_utils import mongolian_tokenizer
    return mongolian_tokenizer(text, vocab, cleaned_text=False)


def synthesize(
    model,
    text: str,
    vocab,
    hps,
    speaker_id: int = 0,
    device: str = "cuda",
) -> tuple:
    tokens = text_to_sequence(text, vocab)
    
    with torch.no_grad():
        x = torch.LongTensor([tokens]).to(device)
        x_lengths = torch.LongTensor([len(tokens)]).to(device)
        sid = torch.LongTensor([speaker_id]).to(device)

        audio = model.infer(x, x_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]
        audio = audio[0, 0].cpu().numpy()

    return audio, hps.data.sample_rate


def main():
    parser = argparse.ArgumentParser(description="Mongolian TTS Inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/mongolian.yaml", help="Config path")
    parser.add_argument("--output", default="output.wav", help="Output audio path")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (0=male, 1=female)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, hps, vocab = load_model(args.checkpoint, args.config, args.device)

    print(f"Synthesizing: {args.text}")
    audio, sr = synthesize(model, args.text, vocab, hps, args.speaker, args.device)

    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
