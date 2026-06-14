from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "giswqs/whu-building-unetplusplus-efficientnet-b4"
FILENAME = "model.pth"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Apache-2.0 Hugging Face WHU building segmentation model."
    )
    parser.add_argument(
        "--output",
        default="models/whu-building-unetplusplus-efficientnet-b4.pth",
        help="Destination path for the model weights.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    output.write_bytes(Path(downloaded).read_bytes())
    print(f"Downloaded {REPO_ID}/{FILENAME} to {output}")


if __name__ == "__main__":
    main()
