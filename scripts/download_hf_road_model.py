from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO_ID = "spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp"
FILENAME = "aerial-image-road-segmentation-xp.keras"
MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}?download=true"
DEFAULT_OUTPUT = Path("models") / FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the MIT-licensed Hugging Face U-Net/Keras road model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing local model file.",
    )
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.force:
        print(f"Model already exists: {output}")
        print("Use --force to download it again.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".part")

    print(f"Downloading {MODEL_URL}")
    print(f"Writing {output}")
    try:
        with urlopen(MODEL_URL) as response, temp_output.open("wb") as file:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    progress = (
                        f"\r{downloaded / 1024 / 1024:,.1f} MB / "
                        f"{total / 1024 / 1024:,.1f} MB ({percent:0.1f}%)"
                    )
                    print(progress, end="")
                else:
                    print(f"\r{downloaded / 1024 / 1024:,.1f} MB", end="")
    except URLError as exc:
        if temp_output.exists():
            temp_output.unlink()
        raise SystemExit(f"\nDownload failed: {exc}") from exc

    temp_output.replace(output)
    print(f"\nDownloaded model to {output}")


if __name__ == "__main__":
    main()
