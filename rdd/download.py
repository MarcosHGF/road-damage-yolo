"""Resumable download and safe extraction of the RDD2022 country archives."""
from __future__ import annotations

import http.client
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .constants import APPROX_ZIP_MB, COUNTRY_ZIPS, Paths
from .util import human_bytes

CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "road-damage-yolo/1.0 (+research; urllib)"


class DownloadError(RuntimeError):
    pass


def _content_range_total(header: str | None) -> int | None:
    # "bytes 100-999/1000" -> 1000
    if header and "/" in header:
        tail = header.rsplit("/", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def download_file(
    url: str, dest: Path, retries: int = 6, timeout: int = 60, quiet: bool = False, backoff: float = 1.0
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` file.

    Network drops are retried with exponential backoff and continue from the
    byte where they stopped, which matters for the multi-GB archives.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempt = 0
    while True:
        pos = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if pos and status != 206:
                    # Server ignored the Range header: start over.
                    pos = 0
                    mode = "wb"
                else:
                    mode = "ab" if pos else "wb"
                total = _content_range_total(resp.headers.get("Content-Range"))
                if total is None:
                    length = resp.headers.get("Content-Length")
                    total = (int(length) + pos) if length and length.isdigit() else None
                done = pos
                last_print = 0.0
                with open(part, mode) as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if not quiet and now - last_print > 0.5:
                            last_print = now
                            _progress(dest.name, done, total)
                if not quiet:
                    _progress(dest.name, done, total)
                    print()
                if total is not None and done != total:
                    raise DownloadError(f"incomplete download: got {done} of {total} bytes")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, DownloadError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and part.exists():
                # Range not satisfiable: the .part file already holds the whole file.
                part.replace(dest)
                return dest
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise DownloadError(
                    f"HTTP {exc.code} for {url}. Download the file in a browser and put it in "
                    f"{dest.parent} (see README, 'Manual download')."
                ) from exc
            attempt += 1
            if attempt > retries:
                raise DownloadError(f"giving up on {url} after {retries} retries: {exc}") from exc
            wait = min(60.0, backoff * 2 ** attempt)
            if not quiet:
                print(f"\n  network error ({exc}); retry {attempt}/{retries} in {wait}s, resuming...")
            time.sleep(wait)


def _progress(name: str, done: int, total: int | None) -> None:
    if total:
        pct = 100 * done / total
        sys.stdout.write(f"\r  {name}: {pct:5.1f}%  {human_bytes(done)} / {human_bytes(total)}   ")
    else:
        sys.stdout.write(f"\r  {name}: {human_bytes(done)}   ")
    sys.stdout.flush()


def verify_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"{path} is not a valid zip file (corrupted or an HTML error page).")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise DownloadError(f"{path} is corrupted (first bad member: {bad}). Delete it and retry.")


def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extract ``zip_path`` into ``dest``, refusing paths that escape it."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise DownloadError(f"unsafe path in archive: {member.filename}")
        zf.extractall(dest)
        count = len(zf.infolist())
    return count


def _find_local_zip(downloads: Path, country: str) -> Path | None:
    for name in (f"{country}.zip", f"RDD2022_{country}.zip"):
        candidate = downloads / name
        if candidate.exists():
            return candidate
    return None


def _check_disk(path: Path, country: str) -> None:
    need_mb = APPROX_ZIP_MB.get(country)
    if not need_mb:
        return
    probe = path
    while not probe.exists():
        probe = probe.parent
    free_mb = shutil.disk_usage(probe).free / (1024 * 1024)
    if free_mb < need_mb * 3:
        print(
            f"  warning: {country} needs roughly {need_mb * 3 / 1024:.1f} GB free "
            f"(archive + extracted + converted copies); only {free_mb / 1024:.1f} GB available."
        )


def fetch_countries(
    countries: list[str],
    paths: Paths,
    keep_zip: bool = False,
    urls: dict[str, str] | None = None,
) -> list[str]:
    """Make sure each country is downloaded and extracted under ``paths.raw``.

    A ``<country>.zip`` (or ``RDD2022_<country>.zip``) already present in
    ``paths.downloads`` is used instead of downloading, which supports
    manual downloads from a browser.
    """
    urls = urls or COUNTRY_ZIPS
    unknown = [c for c in countries if c not in urls]
    if unknown:
        raise DownloadError(f"unknown countries {unknown}; choose from {sorted(urls)}")

    ready: list[str] = []
    for country in countries:
        target = paths.raw / country
        marker = target / ".extracted_ok"
        if marker.exists():
            print(f"[{country}] already extracted, skipping (delete {marker} to redo)")
            ready.append(country)
            continue

        zip_path = _find_local_zip(paths.downloads, country)
        downloaded_here = False
        if zip_path is None:
            _check_disk(paths.data_dir, country)
            zip_path = paths.downloads / f"{country}.zip"
            print(f"[{country}] downloading {urls[country]}")
            download_file(urls[country], zip_path)
            downloaded_here = True
        else:
            print(f"[{country}] using local archive {zip_path}")

        print(f"[{country}] verifying archive...")
        verify_zip(zip_path)
        print(f"[{country}] extracting to {target} ...")
        n = safe_extract(zip_path, target)
        marker.write_text("ok", encoding="utf-8")
        print(f"[{country}] extracted {n} entries")
        if downloaded_here and not keep_zip:
            zip_path.unlink()
        ready.append(country)
    return ready
