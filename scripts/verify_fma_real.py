#!/usr/bin/env python3
"""Validate every selected FMA MP3 with ffprobe and write a CSV report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "real_track_map.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fma_real",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "fma_real_validation.csv",
    )
    args = parser.parse_args()

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")

    with args.mapping.open(encoding="utf-8", newline="") as handle:
        mapping = list(csv.DictReader(handle))

    report: list[dict[str, object]] = []
    for index, row in enumerate(mapping, start=1):
        track_id = int(row["fma_track_id"])
        audio_path = args.audio_dir / f"{track_id:06d}.mp3"
        record: dict[str, object] = {
            "original_audio": row["original_audio"],
            "fma_track_id": track_id,
            "path": str(audio_path),
            "exists": audio_path.is_file(),
            "size_bytes": audio_path.stat().st_size if audio_path.is_file() else 0,
            "format_name": "",
            "codec_name": "",
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "decodable": False,
            "usable_10s": False,
            "error": "",
        }
        if audio_path.is_file():
            command = [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(audio_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode == 0:
                try:
                    payload = json.loads(completed.stdout)
                    stream = payload.get("streams", [{}])[0]
                    audio_format = payload.get("format", {})
                    duration = float(audio_format.get("duration", 0.0))
                    record.update(
                        {
                            "format_name": audio_format.get("format_name", ""),
                            "codec_name": stream.get("codec_name", ""),
                            "sample_rate": stream.get("sample_rate", ""),
                            "channels": stream.get("channels", ""),
                            "duration_sec": round(duration, 6),
                            "decodable": bool(stream.get("codec_name")) and duration > 0,
                            "usable_10s": bool(stream.get("codec_name")) and duration >= 10.0,
                        }
                    )
                except (ValueError, IndexError, TypeError) as error:
                    record["error"] = f"invalid ffprobe JSON: {error}"
            else:
                record["error"] = completed.stderr.strip().replace("\n", " | ")
        else:
            record["error"] = "missing file"
        report.append(record)
        if index % 50 == 0 or index == len(mapping):
            print(f"Validated {index}/{len(mapping)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    durations = [float(row["duration_sec"]) for row in report if row["duration_sec"] != ""]
    print(f"Files expected: {len(report)}")
    print(f"Files present:  {sum(row['exists'] is True for row in report)}")
    print(f"Decodable:      {sum(row['decodable'] is True for row in report)}")
    print(f"Usable >=10s:  {sum(row['usable_10s'] is True for row in report)}")
    print(f"Invalid/short:  {sum(row['usable_10s'] is False for row in report)}")
    if durations:
        print(f"Duration range: {min(durations):.3f} to {max(durations):.3f} sec")
    print(
        "Sample rates:   "
        + str(dict(Counter(str(row["sample_rate"]) for row in report if row["sample_rate"])))
    )
    print(f"Report:         {args.output}")


if __name__ == "__main__":
    main()
