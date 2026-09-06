from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def ensure_raw_dir() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def ensure_processed_dir() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


def ensure_checkpoints_dir() -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS_DIR


def mordred_cache_path() -> Path:
    return PROCESSED_DIR / "mordred_cache.sqlite"
