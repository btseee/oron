"""
Training launcher for Mongolian VITS2 model.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VITS2_DIR = PROJECT_ROOT / "src" / "vits2"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VITS2_DIR))
os.chdir(VITS2_DIR)


def train(
    config: str = "../../configs/mongolian.yaml",
    model_dir: str = "../../checkpoints/mongolian_vits2",
    resume: bool = False,
):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    
    cmd_args = [
        "--config", config,
        "--model", model_dir,
    ]
    
    sys.argv = ["train_ms.py"] + cmd_args
    
    import train_ms
    train_ms.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mongolian VITS2")
    parser.add_argument(
        "--config",
        default="../../configs/mongolian.yaml",
        help="Config file path (relative to src/vits2)",
    )
    parser.add_argument(
        "--model-dir",
        default="../../checkpoints/mongolian_vits2",
        help="Model checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    args = parser.parse_args()
    train(config=args.config, model_dir=args.model_dir, resume=args.resume)
