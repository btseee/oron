# Oron - Mongolian TTS

VITS2-based Text-to-Speech for Mongolian (Khalkha Cyrillic).

## Setup

```bash
git clone --recursive https://github.com/yourusername/oron.git
cd oron
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

```bash
# Prepare dataset
python prepare.py --denoise

# Train
python train.py

# Inference
python inference.py --text "сайн байна уу" --output output.wav

# Upload to HuggingFace
python upload.py --model --dataset
```

## Dataset

Download [Common Voice Mongolian](https://commonvoice.mozilla.org/) and place in:

```
data/raw/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn/
```

## Requirements

- Python 3.10+
- CUDA 11.8+
- FFmpeg

## License

MIT
