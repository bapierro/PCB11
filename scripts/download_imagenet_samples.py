import argparse
from pathlib import Path

from pcb11.imagenet_samples import download_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small set of ImageNet sample images for demos."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/imagenet_samples"),
        help="Where to store downloaded images",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    downloaded = download_samples(output_dir)
    for path in downloaded:
        print(f"Ready {path.name}")


if __name__ == "__main__":
    main()
