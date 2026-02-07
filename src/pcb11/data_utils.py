import shutil
from pathlib import Path


def ensure_image_dir(image_path: Path, working_dir: Path) -> Path:
    if image_path.is_dir():
        return image_path

    working_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = working_dir / "_single_image"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, temp_dir / image_path.name)
    return temp_dir
