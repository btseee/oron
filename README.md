# Oron - Mongolian VITS2 Text-to-Speech

A complete pipeline for training VITS2 text-to-speech models on Mongolian (Khalkha) language using the Mozilla Common Voice dataset.

## Features

- **Audio Processing Pipeline**: Denoising (DeepFilterNet), resampling, trimming, normalization
- **Mongolian Text Processing**: Custom cleaners for Cyrillic text and number normalization
- **Multi-Speaker Support**: Male and female voice support
- **Training Pipeline**: Optimized VITS2 configuration for Mongolian
- **Hugging Face Integration**: Easy model upload and sharing

## Quick Start

### 1. Clone and Setup

```bash
git clone --recurse-submodules https://github.com/btseee/oron.git
cd oron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install espeak-ng for phonemization (optional, for other languages)
# Ubuntu: sudo apt-get install espeak-ng
# Mac: brew install espeak-ng
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Prepare Dataset

If you already have Common Voice Mongolian dataset:

```bash
# Process audio files (denoise, resample, normalize)
python scripts/prepare_dataset.py \
    --data-dir data/processed/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn \
    --output-dir data/prepared
```

Or download fresh:

```bash
# Download Common Voice Mongolian
python scripts/download_dataset.py --output-dir data/raw

# Then prepare
python scripts/prepare_dataset.py \
    --data-dir data/raw/cv-corpus-24.0-2025-12-05/mn \
    --output-dir data/prepared
```

### 4. Train Model

```bash
python scripts/train.py --config datasets/mongolian_base/config.yaml
```

### 5. Inference

```bash
python scripts/inference.py \
    --text "Сайн байна уу" \
    --checkpoint checkpoints/mongolian_vits2/G_100000.pth \
    --output output.wav \
    --speaker 0  # 0=male, 1=female
```

### 6. Upload to Hugging Face

```bash
python scripts/upload_to_huggingface.py \
    --model-dir checkpoints/mongolian_vits2 \
    --repo-id yourusername/mongolian-tts \
    --create-card
```

## Project Structure

```
oron/
├── configs/
│   ├── mongolian.yaml          # VITS2 config for Mongolian
│   └── vocab.txt               # Mongolian vocabulary
├── data/
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Common Voice data
│   └── prepared/               # Processed and ready for training
│       ├── clips/              # Processed audio files
│       └── filelists/          # Train/val/test splits
├── scripts/
│   ├── download_dataset.py     # Download Common Voice
│   ├── audio_processing.py     # Audio denoising/processing
│   ├── metadata_processing.py  # Create filelists
│   ├── prepare_dataset.py      # Full pipeline
│   ├── train.py                # Training launcher
│   ├── inference.py            # TTS inference
│   └── upload_to_huggingface.py
├── src/
│   ├── mongolian/              # Mongolian text processing
│   │   ├── cleaners.py         # Text cleaners
│   │   ├── symbols.py          # Cyrillic symbols
│   │   └── data_utils.py       # Data utilities
│   └── vits2/                  # VITS2 submodule (don't modify)
├── checkpoints/                # Trained models
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

## Audio Processing Pipeline

| Step | Tool | Description |
|------|------|-------------|
| Denoising | DeepFilterNet | Remove background noise |
| Format Conversion | FFmpeg | MP3 to WAV conversion |
| Resampling | librosa | Convert to 22050 Hz |
| Trimming | librosa | Remove silence |
| Normalization | Custom | Normalize amplitude |

## Configuration

Edit `configs/mongolian.yaml`:

```yaml
train:
  batch_size: 32        # Reduce if OOM
  epochs: 10000
  learning_rate: 0.0002

data:
  sample_rate: 22050
  n_speakers: 2         # Update based on your data

model:
  hidden_channels: 192
  n_layers: 6
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~10GB disk space for processed data

## Training Tips

1. **Start small**: Use `--no-denoise` for faster initial testing
2. **Monitor**: Check TensorBoard at `checkpoints/mongolian_vits2`
3. **Batch size**: Reduce if running out of GPU memory
4. **Quality filter**: Adjust `--min-upvotes` for data quality

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [VITS2](https://github.com/daniilrobnikov/vits2) - Original implementation
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Mongolian dataset
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Audio denoising
