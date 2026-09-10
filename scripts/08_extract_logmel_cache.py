#!/usr/bin/env python3
"""7단계-A: CNN 입력용 128-band Log-Mel 캐시를 만든다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24_000
SEGMENT_DURATION_SEC = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION_SEC)
N_FFT = 1024
HOP_LENGTH = 240
N_MELS = 128
TIME_POOL = 4
EXPECTED_STFT_FRAMES = 996
OUTPUT_TIME_BINS = EXPECTED_STFT_FRAMES // TIME_POOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "features"
        / "logmel_128x249_mp3_64kbps.npy",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--codec-control", choices=["none", "mp3"], default="mp3")
    parser.add_argument("--mp3-bitrate-kbps", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_ffmpeg(command: list[str], input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message)
    return result.stdout


def fixed_length(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) < TARGET_SAMPLES:
        waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))
    elif len(waveform) > TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    if len(waveform) != TARGET_SAMPLES or not np.isfinite(waveform).all():
        raise RuntimeError("유효하지 않은 고정 길이 파형")
    return waveform.astype(np.float32, copy=False)


def decode_segment(ffmpeg: str, row: dict[str, str]) -> np.ndarray:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(row['start_sec']):.6f}",
        "-i",
        row["absolute_path"],
        "-t",
        f"{float(row['duration_sec']):.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    return fixed_length(np.frombuffer(run_ffmpeg(command), dtype="<f4").copy())


def mp3_roundtrip(ffmpeg: str, waveform: np.ndarray, bitrate_kbps: int) -> np.ndarray:
    encode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-map_metadata",
        "-1",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        "-f",
        "mp3",
        "pipe:1",
    ]
    encoded = run_ffmpeg(
        encode_command,
        np.asarray(waveform, dtype="<f4").tobytes(),
    )
    decode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    return fixed_length(
        np.frombuffer(run_ffmpeg(decode_command, encoded), dtype="<f4").copy()
    )


def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((N_MELS, len(frequencies)), dtype=np.float32)
    for index in range(N_MELS):
        left, center, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    filters *= (
        2.0 / np.maximum(hz_points[2:] - hz_points[:-2], 1e-12)
    )[:, np.newaxis]
    return filters


def extract_logmel(waveform: np.ndarray, filters: np.ndarray) -> np.ndarray:
    _, _, stft = signal.stft(
        waveform,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    if stft.shape[1] != EXPECTED_STFT_FRAMES:
        raise RuntimeError(f"예상 STFT frame {EXPECTED_STFT_FRAMES}, 실제 {stft.shape[1]}")
    power = np.abs(stft) ** 2
    mel_power = np.maximum(filters @ power, 1e-10)
    # 996 frame을 네 칸씩 선형 power에서 평균하여 CNN 계산량을 줄인다.
    pooled = mel_power.reshape(N_MELS, OUTPUT_TIME_BINS, TIME_POOL).mean(axis=2)
    logmel = 10.0 * np.log10(np.maximum(pooled, 1e-10))
    logmel -= np.max(logmel)
    logmel = np.clip(logmel, -80.0, 0.0)
    if logmel.shape != (N_MELS, OUTPUT_TIME_BINS) or not np.isfinite(logmel).all():
        raise RuntimeError(f"Log-Mel 결과 오류: {logmel.shape}")
    return logmel.astype(np.float16)


def process_one(
    index: int,
    row: dict[str, str],
    ffmpeg: str,
    filters: np.ndarray,
    codec_control: str,
    bitrate_kbps: int,
) -> tuple[int, np.ndarray]:
    try:
        waveform = decode_segment(ffmpeg, row)
        if codec_control == "mp3":
            waveform = mp3_roundtrip(ffmpeg, waveform, bitrate_kbps)
        return index, extract_logmel(waveform, filters)
    except Exception as exc:
        raise RuntimeError(f"{row['segment_id']}: {exc}") from exc


def main() -> None:
    args = parse_args()
    rows = read_csv(args.segment_manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit은 양수여야 합니다.")
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("처리할 세그먼트가 없습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    cache = np.lib.format.open_memmap(
        temporary_output,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), N_MELS, OUTPUT_TIME_BINS),
    )
    filters = mel_filterbank()
    condition = (
        "original"
        if args.codec_control == "none"
        else f"mp3_{args.mp3_bitrate_kbps}kbps_roundtrip"
    )
    started = time.monotonic()
    print(
        f"Log-Mel 추출 시작: {len(rows):,}개, shape=({N_MELS}, {OUTPUT_TIME_BINS}), "
        f"workers={args.workers}, condition={condition}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                row,
                args.ffmpeg,
                filters,
                args.codec_control,
                args.mp3_bitrate_kbps,
            ): index
            for index, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            index, logmel = future.result()
            cache[index] = logmel
            completed += 1
            if completed % 250 == 0 or completed == len(rows):
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(rows) - completed) / max(rate, 1e-9)
                print(
                    f"진행 {completed:,}/{len(rows):,} ({completed / len(rows):.1%}), "
                    f"예상 남은 시간 {remaining:.0f}초",
                    flush=True,
                )
    cache.flush()
    del cache
    os.replace(temporary_output, args.output)

    segment_ids = [row["segment_id"] for row in rows]
    metadata = {
        "shape": [len(rows), N_MELS, OUTPUT_TIME_BINS],
        "dtype": "float16",
        "sample_rate": SAMPLE_RATE,
        "segment_duration_sec": SEGMENT_DURATION_SEC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "time_pool": TIME_POOL,
        "value_min_db": -80.0,
        "value_max_db": 0.0,
        "codec_condition": condition,
        "segment_id_sha256": hashlib.sha256("\n".join(segment_ids).encode()).hexdigest(),
        "segment_count": len(segment_ids),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    print(f"Log-Mel 캐시 완료: {metadata['shape']}, 소요 {elapsed:.1f}초")
    print(f"저장 위치: {args.output}")


if __name__ == "__main__":
    main()
