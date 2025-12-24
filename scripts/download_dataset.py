"""Download Common Voice Mongolian dataset."""

import os
import subprocess
import argparse
from pathlib import Path


DATASET_URL = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-24.0-2025-12-05/cv-corpus-24.0-2025-12-05-mn.tar.gz"


def download_dataset(output_dir: str, extract: bool = True) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    archive_name = "cv-corpus-mn.tar.gz"
    archive_path = output_path / archive_name

    if not archive_path.exists():
        print(f"Downloading Common Voice Mongolian dataset...")
        print(f"Note: You may need to accept the license at commonvoice.mozilla.org")
        print(f"Downloading to: {archive_path}")
        
        try:
            subprocess.run(
                ["wget", "-O", str(archive_path), DATASET_URL],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("wget failed, trying curl...")
            subprocess.run(
                ["curl", "-L", "-o", str(archive_path), DATASET_URL],
                check=True,
            )
    else:
        print(f"Archive already exists: {archive_path}")

    if extract:
        print(f"Extracting archive...")
        subprocess.run(
            ["tar", "-xzf", str(archive_path), "-C", str(output_path)],
            check=True,
        )
        print(f"Extracted to: {output_path}")

    return str(archive_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Common Voice Mongolian dataset")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract the archive",
    )

    args = parser.parse_args()
    download_dataset(args.output_dir, extract=not args.no_extract)
