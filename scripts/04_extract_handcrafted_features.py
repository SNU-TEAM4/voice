#!/usr/bin/env python3
"""4단계-A: 10초 구간에서 MFCC와 기본 음향 특징 266개를 추출한다."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 여러 ffmpeg 작업과 BLAS 내부 스레드가 겹쳐 느려지는 것을 방지한다.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import fft, signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 24_000
SEGMENT_DURATION_SEC = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION_SEC)
N_FFT = 1024
HOP_LENGTH = 240
N_MELS = 64
N_MFCC = 40


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
        default=PROJECT_ROOT / "artifacts" / "features" / "handcrafted_features.npz",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument(
        "--codec-control",
        choices=["none", "mp3"],
        default="none",
        help="mp3를 고르면 공통 비트레이트 MP3 round-trip 후 특징을 추출합니다.",
    )
    parser.add_argument(
        "--mp3-bitrate-kbps",
        type=int,
        default=64,
        help="--codec-control mp3에서 모든 구간에 공통 적용할 비트레이트입니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="파이프라인 시험용으로 앞에서 N개 구간만 처리합니다.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{row['segment_id']}: ffmpeg 실패: {message}")
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(waveform) < TARGET_SAMPLES:
        waveform = np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))
    elif len(waveform) > TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    if len(waveform) != TARGET_SAMPLES or not np.isfinite(waveform).all():
        raise RuntimeError(f"{row['segment_id']}: 유효하지 않은 파형")
    return waveform


def mp3_roundtrip(ffmpeg: str, waveform: np.ndarray, bitrate_kbps: int) -> np.ndarray:
    """고정 길이 PCM을 공통 MP3로 인코딩한 뒤 다시 PCM으로 디코딩한다."""
    if bitrate_kbps <= 0:
        raise ValueError("MP3 비트레이트는 양수여야 합니다.")
    pcm_bytes = np.asarray(waveform, dtype="<f4").tobytes()
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
    encoded = subprocess.run(
        encode_command,
        input=pcm_bytes,
        capture_output=True,
        check=False,
    )
    if encoded.returncode != 0:
        message = encoded.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"공통 MP3 인코딩 실패: {message}")

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
    decoded = subprocess.run(
        decode_command,
        input=encoded.stdout,
        capture_output=True,
        check=False,
    )
    if decoded.returncode != 0:
        message = decoded.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"공통 MP3 디코딩 실패: {message}")
    controlled = np.frombuffer(decoded.stdout, dtype="<f4").copy()

    # MP3 프레임 단위 패딩으로 길이가 조금 늘거나 줄 수 있어 모델 입력 길이에 맞춘다.
    if len(controlled) < TARGET_SAMPLES:
        controlled = np.pad(controlled, (0, TARGET_SAMPLES - len(controlled)))
    elif len(controlled) > TARGET_SAMPLES:
        controlled = controlled[:TARGET_SAMPLES]
    if len(controlled) != TARGET_SAMPLES or not np.isfinite(controlled).all():
        raise RuntimeError("MP3 round-trip 후 유효하지 않은 파형")
    return controlled


def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def make_mel_filterbank() -> np.ndarray:
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((N_MELS, len(frequencies)), dtype=np.float32)
    for index in range(N_MELS):
        left, center, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    enorm = 2.0 / np.maximum(hz_points[2:] - hz_points[:-2], 1e-12)
    filters *= enorm[:, np.newaxis]
    return filters


def summarize_matrix(
    prefix: str,
    values: np.ndarray,
    feature_names: list[str],
    feature_values: list[float],
) -> None:
    if values.ndim == 1:
        values = values[np.newaxis, :]
    for index, row in enumerate(values, start=1):
        feature_names.extend([f"{prefix}_{index:02d}_mean", f"{prefix}_{index:02d}_std"])
        feature_values.extend([float(np.mean(row)), float(np.std(row))])


def extract_feature_vector(
    waveform: np.ndarray,
    mel_filters: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    frequencies, _, stft = signal.stft(
        waveform,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    magnitude = np.maximum(np.abs(stft), 1e-10)
    power = magnitude**2
    mel_power = np.maximum(mel_filters @ power, 1e-10)
    log_mel = np.log(mel_power)
    mfcc = fft.dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    mfcc_delta = np.gradient(mfcc, axis=1)
    mfcc_delta2 = np.gradient(mfcc_delta, axis=1)

    feature_names: list[str] = []
    feature_values: list[float] = []
    summarize_matrix("mfcc", mfcc, feature_names, feature_values)
    summarize_matrix("mfcc_delta", mfcc_delta, feature_names, feature_values)
    summarize_matrix("mfcc_delta2", mfcc_delta2, feature_names, feature_values)

    magnitude_sum = np.maximum(np.sum(magnitude, axis=0), 1e-12)
    centroid = np.sum(frequencies[:, None] * magnitude, axis=0) / magnitude_sum
    bandwidth = np.sqrt(
        np.sum(((frequencies[:, None] - centroid[None, :]) ** 2) * magnitude, axis=0)
        / magnitude_sum
    )
    cumulative_power = np.cumsum(power, axis=0)
    rolloff_target = 0.85 * cumulative_power[-1]
    rolloff_indices = np.argmax(cumulative_power >= rolloff_target[None, :], axis=0)
    rolloff = frequencies[rolloff_indices]
    flatness = np.exp(np.mean(np.log(magnitude), axis=0)) / np.maximum(
        np.mean(magnitude, axis=0), 1e-12
    )

    frames = np.lib.stride_tricks.sliding_window_view(waveform, N_FFT)[::HOP_LENGTH]
    frame_count = min(frames.shape[0], magnitude.shape[1])
    frames = frames[:frame_count]
    magnitude = magnitude[:, :frame_count]
    centroid = centroid[:frame_count]
    bandwidth = bandwidth[:frame_count]
    rolloff = rolloff[:frame_count]
    flatness = flatness[:frame_count]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)

    scalar_features = {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff85": rolloff,
        "spectral_flatness": flatness,
        "rms": rms,
        "zcr": zcr,
    }
    for name, values in scalar_features.items():
        summarize_matrix(name, values, feature_names, feature_values)

    # 0~12kHz를 7개 대역으로 나누고 각 대역의 강약 대비를 계산한다.
    contrast_edges = np.asarray([0, 200, 400, 800, 1600, 3200, 6400, 12001])
    log_magnitude = 20.0 * np.log10(magnitude)
    contrasts = []
    for lower, upper in zip(contrast_edges[:-1], contrast_edges[1:]):
        mask = (frequencies >= lower) & (frequencies < upper)
        band = log_magnitude[mask]
        if band.shape[0] < 2:
            contrasts.append(np.zeros(frame_count))
        else:
            contrasts.append(np.percentile(band, 90, axis=0) - np.percentile(band, 10, axis=0))
    summarize_matrix(
        "spectral_contrast",
        np.asarray(contrasts),
        feature_names,
        feature_values,
    )

    vector = np.asarray(feature_values, dtype=np.float32)
    if len(vector) != 266 or not np.isfinite(vector).all():
        raise RuntimeError(f"특징 벡터 오류: shape={vector.shape}, finite={np.isfinite(vector).all()}")
    return vector, feature_names


def process_one(
    index: int,
    row: dict[str, str],
    ffmpeg: str,
    mel_filters: np.ndarray,
    codec_control: str,
    mp3_bitrate_kbps: int,
) -> tuple[int, np.ndarray, list[str]]:
    waveform = decode_segment(ffmpeg, row)
    if codec_control == "mp3":
        waveform = mp3_roundtrip(ffmpeg, waveform, mp3_bitrate_kbps)
    vector, names = extract_feature_vector(waveform, mel_filters)
    return index, vector, names


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
    mel_filters = make_mel_filterbank()
    vectors: list[np.ndarray | None] = [None] * len(rows)
    feature_names: list[str] | None = None
    started = time.monotonic()

    condition = (
        "original"
        if args.codec_control == "none"
        else f"mp3_{args.mp3_bitrate_kbps}kbps_roundtrip"
    )
    print(
        f"특징 추출 시작: {len(rows):,}개, workers={args.workers}, condition={condition}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                row,
                args.ffmpeg,
                mel_filters,
                args.codec_control,
                args.mp3_bitrate_kbps,
            ): index
            for index, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            index, vector, names = future.result()
            vectors[index] = vector
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("세그먼트마다 특징 이름 순서가 다릅니다.")
            completed += 1
            if completed % 250 == 0 or completed == len(rows):
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(rows) - completed) / max(rate, 1e-9)
                print(
                    f"진행 {completed:,}/{len(rows):,} "
                    f"({completed / len(rows):.1%}), 예상 남은 시간 {remaining:.0f}초",
                    flush=True,
                )

    if feature_names is None or any(vector is None for vector in vectors):
        raise RuntimeError("특징 추출 결과가 완성되지 않았습니다.")
    matrix = np.stack(vectors).astype(np.float32)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int8)
    segment_ids = np.asarray([row["segment_id"] for row in rows])
    if matrix.shape != (len(rows), 266):
        raise RuntimeError(f"예상하지 못한 특징 행렬 크기: {matrix.shape}")

    np.savez_compressed(
        args.output,
        X=matrix,
        y=labels,
        segment_ids=segment_ids,
        feature_names=np.asarray(feature_names),
        sample_rate=np.asarray(SAMPLE_RATE),
        segment_duration_sec=np.asarray(SEGMENT_DURATION_SEC),
        codec_condition=np.asarray(condition),
    )
    elapsed = time.monotonic() - started
    print(f"특징 추출 완료: X={matrix.shape}, finite={np.isfinite(matrix).all()}")
    print(f"소요 시간: {elapsed:.1f}초")
    print(f"저장 위치: {args.output}")


if __name__ == "__main__":
    main()
