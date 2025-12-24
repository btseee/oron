#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VITS2_DIR = Path("src/vits2")
CONFIG_PATH = Path("configs/config.yaml")


def train(resume: bool = False):
    model_dir = Path(os.getenv("MODEL_DIR", "checkpoints/oron"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    
    cmd = [
        sys.executable,
        str(VITS2_DIR / "train_ms.py"),
        "-c", str(CONFIG_PATH.absolute()),
        "-m", str(model_dir.absolute()),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=str(VITS2_DIR))


def main():
    parser = argparse.ArgumentParser(description="Train VITS2 model")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    train(resume=args.resume)


if __name__ == "__main__":
    main()
