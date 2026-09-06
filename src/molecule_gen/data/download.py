from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES = 16 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def format_bytes(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_with_resume(url: str, dest: Path) -> None:
    """Download url to dest, resuming a partial file if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        _download_with_curl(url, dest)
        return
    _download_with_urllib(url, dest)


def _download_with_curl(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {dest.name} via curl (resume-safe)")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "--output",
        str(part),
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        raise DownloadError(
            f"curl failed to download {url} (exit {result.returncode}). "
            f"You can save the file yourself to {dest}."
        )
    part.replace(dest)
    log(f"downloaded {dest.name} ({format_bytes(dest.stat().st_size)})")


def _download_with_urllib(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "molecule-gen/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        log(f"resuming {dest.name} from {format_bytes(existing)}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if existing and exc.code == 416:
            part.replace(dest)
            return
        if existing and exc.code == 200:
            existing = 0
            part.unlink(missing_ok=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "molecule-gen/0.1"}
            )
            response = urllib.request.urlopen(request, timeout=120)
        else:
            raise DownloadError(f"Failed to download {url}: {exc}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    status = getattr(response, "status", None) or response.getcode()
    content_range = response.headers.get("Content-Range", "")
    total_header = response.headers.get("Content-Length")

    if existing and status == 200 and not content_range:
        existing = 0
        part.unlink(missing_ok=True)

    if content_range and "/" in content_range:
        try:
            total = int(content_range.rsplit("/", 1)[1])
        except ValueError:
            total = existing + int(total_header) if total_header else None
    elif total_header:
        total = existing + int(total_header)
    else:
        total = None

    mode = "ab" if existing else "wb"
    downloaded = existing
    last_report = existing
    try:
        with part.open(mode) as handle:
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if downloaded - last_report >= PROGRESS_EVERY_BYTES:
                    if total:
                        pct = 100.0 * downloaded / total
                        log(
                            f"downloading {dest.name}: {format_bytes(downloaded)} / "
                            f"{format_bytes(total)} ({pct:.1f}%)"
                        )
                    else:
                        log(f"downloading {dest.name}: {format_bytes(downloaded)}")
                    last_report = downloaded
    finally:
        response.close()

    part.replace(dest)
    log(f"downloaded {dest.name} ({format_bytes(downloaded)})")


def find_local_file(name: str, *folders: Path) -> Path | None:
    seen: set[Path] = set()
    for folder in folders:
        path = folder / name
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        return path
    return None
