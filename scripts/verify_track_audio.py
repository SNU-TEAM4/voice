#!/usr/bin/env python3
"""Validate every REAL/TTA track in the integrated manifest with ffprobe."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def inspect(row: dict[str, str], ffprobe: str) -> dict[str, object]:
    path = Path(row["absolute_path"])
    record: dict[str, object] = {
        "sample_id": row["sample_id"],
        "group_id": row["group_id"],
        "label": row["label"],
        "generator": row["generator"],
        "split": row["split"],
        "absolute_path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "format_name": "",
        "codec_name": "",
        "sample_rate": "",
        "channels": "",
        "metadata_duration_sec": row["duration_sec"],
        "actual_duration_sec": "",
        "duration_delta_sec": "",
        "decodable": False,
        "usable_10s": False,
        "error": "",
    }
    if not path.is_file():
        record["error"] = "missing file"
        return record

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=duration,format_name:stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        record["error"] = "ffprobe timeout after 30 seconds"
        return record
    if completed.returncode != 0:
        record["error"] = completed.stderr.strip().replace("\n", " | ")
        return record

    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise ValueError("no audio stream")
        stream = streams[0]
        audio_format = payload.get("format", {})
        duration = float(audio_format.get("duration", 0.0))
        metadata_duration = float(row["duration_sec"])
        codec = stream.get("codec_name", "")
        record.update(
            {
                "format_name": audio_format.get("format_name", ""),
                "codec_name": codec,
                "sample_rate": stream.get("sample_rate", ""),
                "channels": stream.get("channels", ""),
                "actual_duration_sec": round(duration, 6),
                "duration_delta_sec": round(duration - metadata_duration, 6),
                "decodable": bool(codec) and duration > 0,
                "usable_10s": bool(codec) and duration >= 10.0,
            }
        )
    except (ValueError, IndexError, TypeError) as error:
        record["error"] = f"invalid ffprobe result: {error}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_audio_validation.csv",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise RuntimeError("Track manifest is empty")

    report: list[dict[str, object] | None] = [None] * len(manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(inspect, row, ffprobe): index
            for index, row in enumerate(manifest)
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            report[futures[future]] = future.result()
            completed_count += 1
            if completed_count % 250 == 0 or completed_count == len(manifest):
                print(f"Validated {completed_count}/{len(manifest)}", flush=True)

    final_report = [row for row in report if row is not None]
    if len(final_report) != len(manifest):
        raise RuntimeError("Validation report row count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_report[0]))
        writer.writeheader()
        writer.writerows(final_report)

    invalid = [row for row in final_report if row["usable_10s"] is False]
    print(f"Tracks expected: {len(final_report)}")
    print(f"Files present:   {sum(row['exists'] is True for row in final_report)}")
    print(f"Decodable:       {sum(row['decodable'] is True for row in final_report)}")
    print(f"Usable >=10s:   {sum(row['usable_10s'] is True for row in final_report)}")
    print(f"Invalid/short:   {len(invalid)}")
    print("Labels:          " + str(dict(Counter(str(row["label"]) for row in final_report))))
    print(
        "Sample rates:    "
        + str(dict(Counter(str(row["sample_rate"]) for row in final_report)))
    )
    print(f"Report:          {args.output}")
    for row in invalid[:20]:
        print(f"INVALID {row['sample_id']}: {row['error']}")


if __name__ == "__main__":
    main()
