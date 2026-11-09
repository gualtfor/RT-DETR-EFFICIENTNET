import argparse
import shutil
import tarfile
from pathlib import Path
from loguru import logger

def setup_dataset(source: str, local_archive: str, destination: str) -> None:
    source = Path(source)
    local_archive = Path(local_archive)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)

    free_space = shutil.disk_usage("/content").free
    archive_size = source.stat().st_size

    logger.info(f"Archive size: {archive_size / 1024**3:.2f} GB")
    logger.info(f"Available space: {free_space / 1024**3:.2f} GB")

    if free_space < archive_size * 1.2:
        raise RuntimeError("Not enough local disk space for the archive copy.")

    logger.info("Copying archive...")
    shutil.copy2(source, local_archive)

    if source.stat().st_size != local_archive.stat().st_size:
        raise RuntimeError("Archive copy size mismatch!")

    logger.info("Archive copied successfully.")

    destination.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting PIDray...")
    with tarfile.open(local_archive, "r:*") as tar:
        tar.extractall(path=destination, filter="data")

    logger.info("Extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--local_archive", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)

    args = parser.parse_args()

    setup_dataset(args.source, args.local_archive, args.destination)
