# Oron - Mongolian TTS

Text-to-Speech model for Mongolian (Khalkha) using VITS2.

## Quick Start

```bash
# Clone
git clone --recursive https://github.com/yourusername/oron.git
cd oron

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your settings

# Prepare dataset
python prepare.py

# Train
python train.py

# Upload to Hugging Face
python upload.py --model --dataset
```

## Dataset

Uses [Mozilla Common Voice](https://commonvoice.mozilla.org/) Mongolian dataset.

Place the dataset in:

```text
data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn/
├── clips/
├── train.tsv
├── dev.tsv
└── test.tsv
```

## Project Structure

```text
oron/
├── prepare.py          # Dataset preparation
├── train.py            # Model training
├── upload.py           # Hugging Face upload
├── configs/
│   └── config.yaml     # VITS2 configuration
├── scripts/
│   ├── audio/          # Audio processing
│   ├── text/           # Text cleaning
│   ├── metadata.py     # Metadata handling
│   └── vocab.py        # Vocabulary generation
├── src/
│   └── vits2/          # VITS2 submodule
└── data/
    ├── raw/            # Original dataset
    └── prepared/       # Processed data
```

## Configuration

### Audio Processing

- Sample rate: 22050 Hz
- Denoising: DeepFilterNet (optional)
- Trimming: librosa silence trim
- Normalization: Peak normalization

### Model

- 2 speakers (male/female)
- Character-level tokenization
- Mongolian Cyrillic alphabet

## Commands

```bash
# Prepare with denoising
python prepare.py --denoise

# Skip audio processing
python prepare.py --skip-audio

# Resume training
python train.py --resume

# Inference
python inference.py --text "Сайн байна уу" --speaker 0 --output output.wav

# Upload model only
python upload.py --model

# Upload dataset only
python upload.py --dataset
```

## Requirements

- Python 3.10+
- CUDA 11.8+
- FFmpeg
- espeak-ng

## License

MIT
