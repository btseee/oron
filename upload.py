#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()


def upload_model():
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_MODEL_REPO_ID", "btsee/oron")
    model_dir = Path(os.getenv("MODEL_DIR", "checkpoints/oron"))
    
    if not token:
        raise ValueError("HF_TOKEN not set in .env")
    
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")
    
    files_to_upload = [
        "G_*.pth",
        "D_*.pth",
        "config.yaml",
    ]
    
    for pattern in files_to_upload:
        for filepath in model_dir.glob(pattern):
            print(f"Uploading {filepath.name}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filepath.name,
                repo_id=repo_id,
            )
    
    print(f"Model uploaded to https://huggingface.co/{repo_id}")


def upload_dataset():
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_DATASET_REPO_ID", "btsee/common-voices-mn")
    data_dir = Path("data/prepared")
    
    if not token:
        raise ValueError("HF_TOKEN not set in .env")
    
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")
    
    for filepath in data_dir.glob("*.csv"):
        print(f"Uploading {filepath.name}...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    for filepath in (data_dir / "filelists").glob("*.txt"):
        print(f"Uploading filelists/{filepath.name}...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"filelists/{filepath.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    vocab_path = data_dir / "vocab.txt"
    if vocab_path.exists():
        api.upload_file(
            path_or_fileobj=str(vocab_path),
            path_in_repo="vocab.txt",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face")
    parser.add_argument("--model", action="store_true", help="Upload model")
    parser.add_argument("--dataset", action="store_true", help="Upload dataset")
    args = parser.parse_args()
    
    if args.model:
        upload_model()
    
    if args.dataset:
        upload_dataset()
    
    if not args.model and not args.dataset:
        print("Specify --model and/or --dataset")


if __name__ == "__main__":
    main()
