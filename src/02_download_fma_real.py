#!/usr/bin/env python3
"""
Download only the selected FMA REAL tracks and reproduce the official FMA 30-second clip rule.

Default inputs (relative to project root):
  data/metadata/fma_real_mapping.csv
  data/raw/FMA/fma_metadata/raw_tracks.csv

Default outputs:
  data/raw/FMA/selected_full/      # downloaded full MP3s
  data/processed/FMA/real_30s/     # official-style middle 30s clips
  data/metadata/fma_real_download_report.csv

Usage examples:
  # Test only 5 tracks first
  python src/02_download_fma_real.py --limit 5

  # Download all 296 tracks
  python src/02_download_fma_real.py

Notes:
- For duration <= 30 s, the source file is copied unchanged.
- For duration > 30 s, start = duration // 2 - 15 and ffmpeg stream-copy is used,
  matching the FMA creation.py logic as closely as possible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BASE_URL = "https://files.freemusicarchive.org/"


def find_project_root() -> Path:
    """Find the project root from cwd or script location."""
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for p in candidates:
        if (p / "data").exists():
            return p.resolve()
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. "
        "project 폴더에서 실행하거나 src/ 안에 이 파일을 두세요."
    )


def parse_duration(value) -> int:
    """Convert FMA raw track_duration (MM:SS or HH:MM:SS) to integer seconds."""
    text = str(value).strip()
    parts = text.split(":")
    try:
        nums = [int(x) for x in parts]
    except ValueError as e:
        raise ValueError(f"잘못된 track_duration 값: {value!r}") from e

    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    raise ValueError(f"지원하지 않는 track_duration 형식: {value!r}")


def fma_audio_path(base_dir: Path, track_id: int) -> Path:
    """Official FMA-like relative layout: 001/001382.mp3."""
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return base_dir / folder / filename


def build_url(track_file: str) -> str:
    """
    Preserve path separators but safely percent-encode spaces/unicode/special chars.
    """
    clean = str(track_file).lstrip("/")
    return BASE_URL + quote(clean, safe="/:@?&=+$,;~-_.!()'")


def download_file(
    session: requests.Session,
    url: str,
    dst: Path,
    timeout: int = 60,
    retries: int = 3,
) -> tuple[bool, str]:
    if dst.exists() and dst.stat().st_size > 0:
        return True, "already_exists"

    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(temp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if temp.stat().st_size == 0:
                raise IOError("다운로드된 파일 크기가 0 byte입니다.")

            temp.replace(dst)
            return True, "downloaded"

        except Exception as e:
            if temp.exists():
                temp.unlink()
            if attempt == retries:
                return False, f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)

    return False, "unknown_error"


def create_fma_clip(src: Path, dst: Path, duration_sec: int) -> tuple[bool, str, int]:
    """
    Reproduce the FMA clip creation rule:
      duration <= 30: copy
      duration > 30: start = duration // 2 - 15, 30 s stream-copy
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        start = 0 if duration_sec <= 30 else duration_sec // 2 - 15
        return True, "already_exists", start

    if duration_sec <= 30:
        try:
            shutil.copyfile(src, dst)
            return True, "copied_full_track", 0
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", 0

    start = duration_sec // 2 - 15

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-ss", str(start),
        "-t", "30",
        "-acodec", "copy",
        str(dst),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        if dst.exists():
            dst.unlink()
        err = result.stderr.strip().splitlines()
        msg = err[-1] if err else "ffmpeg failed"
        return False, msg, start

    if not dst.exists() or dst.stat().st_size == 0:
        return False, "ffmpeg 결과 파일이 생성되지 않았습니다.", start

    return True, "trimmed_middle_30s", start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서 N곡만 테스트합니다. 예: --limit 5",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP 요청 timeout(초), 기본 60",
    )
    parser.add_argument(
        "--skip-clips",
        action="store_true",
        help="원본 다운로드만 하고 30초 clip 생성은 생략",
    )
    args = parser.parse_args()

    project_root = find_project_root()

    mapping_path = project_root / "data/metadata/fma_real_mapping.csv"
    raw_tracks_path = project_root / "data/raw/FMA/fma_metadata/raw_tracks.csv"
    full_dir = project_root / "data/raw/FMA/selected_full"
    clip_dir = project_root / "data/processed/FMA/real_30s"
    report_path = project_root / "data/metadata/fma_real_download_report.csv"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"{mapping_path} 가 없습니다.\n"
            "먼저 02_fma_matching_check.ipynb에서 fma_real_mapping.csv를 저장하세요."
        )
    if not raw_tracks_path.exists():
        raise FileNotFoundError(f"{raw_tracks_path} 가 없습니다.")

    if not args.skip_clips and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. "
            "Mac에서는 `brew install ffmpeg` 후 다시 실행하세요."
        )

    mapping = pd.read_csv(mapping_path)
    raw_tracks = pd.read_csv(raw_tracks_path, index_col=0)
    raw_tracks.index = raw_tracks.index.astype(int)

    required_cols = {"track_id", "original_audio", "genre"}
    missing_cols = required_cols - set(mapping.columns)
    if missing_cols:
        raise ValueError(f"mapping CSV에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    work = mapping.copy()
    work["track_id"] = work["track_id"].astype(int)

    if args.limit is not None:
        work = work.head(args.limit).copy()

    missing_raw = sorted(set(work["track_id"]) - set(raw_tracks.index))
    if missing_raw:
        raise ValueError(f"raw_tracks.csv에 없는 track_id: {missing_raw[:20]}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 FMA-Research-Downloader/1.0"
    })

    results = []

    print(f"Project root : {project_root}")
    print(f"Tracks       : {len(work)}")
    print(f"Full audio   : {full_dir}")
    print(f"30s clips    : {clip_dir}")
    print()

    for i, row in enumerate(work.itertuples(index=False), start=1):
        tid = int(row.track_id)
        raw = raw_tracks.loc[tid]

        track_file = raw["track_file"]
        duration_text = raw["track_duration"]
        duration_sec = parse_duration(duration_text)
        url = build_url(track_file)

        full_path = fma_audio_path(full_dir, tid)
        clip_path = fma_audio_path(clip_dir, tid)

        print(f"[{i:03d}/{len(work):03d}] track_id={tid} | {row.original_audio}")

        ok_download, download_status = download_file(
            session=session,
            url=url,
            dst=full_path,
            timeout=args.timeout,
        )

        clip_ok = False
        clip_status = "skipped"
        clip_start_sec = None

        if ok_download and not args.skip_clips:
            clip_ok, clip_status, clip_start_sec = create_fma_clip(
                src=full_path,
                dst=clip_path,
                duration_sec=duration_sec,
            )

        print(f"  download: {download_status}")
        if not args.skip_clips:
            print(f"  clip    : {clip_status}")

        results.append({
            "track_id": tid,
            "original_audio": row.original_audio,
            "genre": row.genre,
            "track_file": track_file,
            "track_duration": duration_text,
            "duration_sec": duration_sec,
            "download_url": url,
            "full_path": str(full_path.relative_to(project_root)),
            "download_ok": ok_download,
            "download_status": download_status,
            "clip_path": (
                str(clip_path.relative_to(project_root))
                if not args.skip_clips else ""
            ),
            "clip_start_sec": clip_start_sec,
            "clip_ok": clip_ok if not args.skip_clips else None,
            "clip_status": clip_status,
        })

        # Keep a continuously updated report, useful if interrupted.
        pd.DataFrame(results).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )

    report = pd.DataFrame(results)

    print("\n===== SUMMARY =====")
    print("Requested       :", len(report))
    print("Download success:", int(report["download_ok"].sum()))
    print("Download failed :", int((~report["download_ok"]).sum()))

    if not args.skip_clips:
        print("Clip success    :", int(report["clip_ok"].fillna(False).sum()))
        print("Clip failed     :", int((~report["clip_ok"].fillna(False)).sum()))

    print("Report          :", report_path)

    failures = report[~report["download_ok"]]
    if len(failures):
        print("\n===== DOWNLOAD FAILURES =====")
        print(
            failures[
                ["track_id", "original_audio", "download_status", "download_url"]
            ].to_string(index=False)
        )

    if not args.skip_clips:
        clip_failures = report[
            report["download_ok"] & ~report["clip_ok"].fillna(False)
        ]
        if len(clip_failures):
            print("\n===== CLIP FAILURES =====")
            print(
                clip_failures[
                    ["track_id", "original_audio", "clip_status"]
                ].to_string(index=False)
            )

    # Non-zero exit when something failed.
    if (~report["download_ok"]).any():
        sys.exit(2)
    if not args.skip_clips and (~report["clip_ok"].fillna(False)).any():
        sys.exit(3)


if __name__ == "__main__":
    main()
