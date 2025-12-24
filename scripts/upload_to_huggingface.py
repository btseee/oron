"""
Upload trained model to Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv


def upload_to_huggingface(
    model_dir: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
):
    load_dotenv()
    
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Set it in .env or pass --token")

    api = HfApi(token=token)

    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Repository check: {e}")

    model_path = Path(model_dir)
    
    files_to_upload = [
        "G_*.pth",
        "D_*.pth",
        "config.yaml",
        "vocab.txt",
    ]

    print(f"Uploading from {model_dir} to {repo_id}...")
    
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        token=token,
        ignore_patterns=["events.*", "*.spec.pt"],
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")


def create_model_card(repo_id: str, token: str = None):
    load_dotenv()
    token = token or os.getenv("HF_TOKEN")
    
    model_card = """---
license: mit
language:
  - mn
tags:
  - text-to-speech
  - tts
  - mongolian
  - vits2
library_name: vits2
---

# Mongolian VITS2 Text-to-Speech Model

This is a VITS2 model trained on Mongolian (Khalkha) speech data from Common Voice.

## Model Description

- **Language:** Mongolian (mn)
- **Architecture:** VITS2
- **Speakers:** Multi-speaker (male/female)
- **Sample Rate:** 22050 Hz

## Usage

```python
from scripts.inference import load_model, synthesize

model, hps, vocab = load_model("checkpoint.pth", "config.yaml")
audio, sr = synthesize(model, "Сайн байна уу", vocab, hps, speaker_id=0)
```

## Training Data

Trained on Mozilla Common Voice Mongolian dataset.

## License

MIT License
"""
    
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )
    print("Model card uploaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--repo-id", help="HF repo ID (default: from .env)")
    parser.add_argument("--token", help="HF token (default: from .env)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--create-card", action="store_true", help="Create model card")

    args = parser.parse_args()
    
    load_dotenv()
    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    
    if not repo_id:
        raise ValueError("Repo ID required. Set HF_REPO_ID in .env or pass --repo-id")

    upload_to_huggingface(
        model_dir=args.model_dir,
        repo_id=repo_id,
        token=args.token,
        private=args.private,
    )

    if args.create_card:
        create_model_card(repo_id, args.token)
