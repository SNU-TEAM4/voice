#!/usr/bin/env python3
"""Download selected FMA clips from fma_large.zip via HTTP byte ranges.

The remote archive is about 93 GiB. This script exposes it as a seekable,
block-cached stream so Python's zipfile module fetches only the central directory
and the selected members listed in real_track_map.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
USER_AGENT = "EchoesCourseProject/1.0"


class HTTPRangeReader(io.RawIOBase):
    def __init__(
        self,
        url: str,
        block_size: int = 1024 * 1024,
        max_cached_blocks: int = 32,
        timeout: int = 60,
        retries: int = 4,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.timeout = timeout
        self.retries = retries
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.ssl_context = self._create_ssl_context()
        self.length = self._get_length()

    @staticmethod
    def _create_ssl_context() -> ssl.SSLContext:
        """Use certifi on python.org macOS builds that lack a default CA path."""
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            system_ca = Path("/etc/ssl/cert.pem")
            if system_ca.is_file():
                return ssl.create_default_context(cafile=str(system_ca))
            return ssl.create_default_context()

    def _request(self, request: urllib.request.Request):
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                    context=self.ssl_context,
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise OSError(f"Network request failed after {self.retries} attempts") from last_error

    def _get_length(self) -> int:
        request = urllib.request.Request(
            self.url,
            method="HEAD",
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        )
        with self._request(request) as response:
            length = response.headers.get("Content-Length")
            accept_ranges = response.headers.get("Accept-Ranges", "").casefold()
        if not length:
            raise OSError("Remote archive did not report Content-Length")
        if "bytes" not in accept_ranges:
            raise OSError("Remote archive does not advertise byte-range support")
        return int(length)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.position + offset
        elif whence == os.SEEK_END:
            target = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if target < 0:
            raise ValueError("Negative seek position")
        self.position = target
        return self.position

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self.cache.pop(block_index, None)
        if cached is not None:
            self.cache[block_index] = cached
            return cached

        start = block_index * self.block_size
        end = min(start + self.block_size, self.length) - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            },
        )
        with self._request(request) as response:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise OSError(
                    f"Expected HTTP 206 for range {start}-{end}, received {status}; "
                    "refusing to download the full 93 GiB archive"
                )
            data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise OSError(
                f"Short range response for {start}-{end}: {len(data)} of {expected} bytes"
            )

        self.cache[block_index] = data
        while len(self.cache) > self.max_cached_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length or size == 0:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
            if size > 64 * 1024 * 1024:
                raise OSError("Refusing an unbounded read larger than 64 MiB")
        size = min(size, self.length - self.position)
        start_position = self.position
        output = bytearray()

        while len(output) < size:
            block_index = self.position // self.block_size
            block = self._fetch_block(block_index)
            within_block = self.position % self.block_size
            take = min(size - len(output), len(block) - within_block)
            if take <= 0:
                raise OSError("Unable to advance while reading remote archive")
            output.extend(block[within_block : within_block + take])
            self.position += take

        if self.position != start_position + size:
            raise OSError("Remote read position mismatch")
        return bytes(output)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def load_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"fma_track_id", "archive_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Mapping must contain columns: {sorted(required)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "real_track_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fma_real",
    )
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = load_mapping(args.mapping)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    remote_probe = HTTPRangeReader(args.archive_url)
    print(
        f"Remote archive: {remote_probe.length / 1024**3:.2f} GiB "
        f"(range access only, {args.workers} workers)",
        flush=True,
    )

    downloaded = 0
    skipped = 0
    total_bytes = 0
    failures: list[str] = []
    thread_state = threading.local()

    def thread_archive() -> zipfile.ZipFile:
        archive = getattr(thread_state, "archive", None)
        if archive is None:
            reader = HTTPRangeReader(args.archive_url)
            archive = zipfile.ZipFile(reader)
            thread_state.reader = reader
            thread_state.archive = archive
        return archive

    def extract(row: dict[str, str]) -> tuple[str, int, str, str]:
        track_id = int(row["fma_track_id"])
        member_name = row["archive_path"]
        destination = args.output_dir / f"{track_id:06d}.mp3"
        temporary = destination.with_suffix(".mp3.part")
        try:
            archive = thread_archive()
            info = archive.getinfo(member_name)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                return ("skipped", info.file_size, destination.name, "")

            if temporary.exists():
                temporary.unlink()
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise OSError(
                    f"Extracted size mismatch: {temporary.stat().st_size} != {info.file_size}"
                )
            temporary.replace(destination)
            return ("downloaded", info.file_size, destination.name, "")
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            return ("failed", 0, destination.name, f"{track_id}: {error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract, row): row for row in rows}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            status, file_size, file_name, error = future.result()
            if status == "downloaded":
                downloaded += 1
                total_bytes += file_size
                detail = f"saved {file_name} ({file_size / 1024**2:.2f} MiB)"
            elif status == "skipped":
                skipped += 1
                total_bytes += file_size
                detail = f"skip {file_name}"
            else:
                failures.append(error)
                detail = f"FAILED {error}"
            print(f"[{completed:03d}/{len(rows):03d}] {detail}", flush=True)

    print(f"Downloaded: {downloaded}", flush=True)
    print(f"Skipped:    {skipped}", flush=True)
    print(f"Failures:   {len(failures)}", flush=True)
    print(f"Local size: {total_bytes / 1024**2:.1f} MiB", flush=True)
    print(f"Output:     {args.output_dir}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
