#!/usr/bin/env python3
"""3단계: 실제 10초 구간 디코딩을 검사하고 Log-Mel 그림을 만든다."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import wave
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE = 24_000
TARGET_DURATION_SEC = 10.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "audio_eda",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"빈 CSV는 저장하지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_segment(
    ffmpeg: str,
    audio_path: str,
    start_sec: float,
    duration_sec: float = TARGET_DURATION_SEC,
) -> tuple[np.ndarray, int, int]:
    """ffmpeg로 mono·24kHz float32 파형을 읽고 정확히 10초로 맞춘다.

    반환값은 (고정 길이 파형, 원래 디코딩된 샘플 수, 패딩 샘플 수)다.
    """
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        audio_path,
        "-t",
        f"{duration_sec:.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 디코딩 실패: {message}")

    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    decoded_samples = len(waveform)
    padding_samples = max(0, TARGET_SAMPLES - decoded_samples)
    if padding_samples:
        waveform = np.pad(waveform, (0, padding_samples))
    if len(waveform) > TARGET_SAMPLES:
        waveform = waveform[:TARGET_SAMPLES]
    if len(waveform) != TARGET_SAMPLES:
        raise RuntimeError(
            f"고정 길이 변환 실패: expected={TARGET_SAMPLES}, actual={len(waveform)}"
        )
    return waveform, decoded_samples, padding_samples


def write_preview_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(frequency_mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(frequency_mel) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    frequency_bins = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SAMPLE_RATE)
    mel_points = np.linspace(
        hz_to_mel(0.0), hz_to_mel(TARGET_SAMPLE_RATE / 2), N_MELS + 2
    )
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((N_MELS, len(frequency_bins)), dtype=np.float32)

    for index in range(N_MELS):
        left, center, right = hz_points[index : index + 3]
        rising = (frequency_bins - left) / max(center - left, 1e-12)
        falling = (right - frequency_bins) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))

    # 주파수 구간이 넓은 필터가 값 자체만으로 유리해지지 않게 면적을 맞춘다.
    enorm = 2.0 / np.maximum(hz_points[2 : N_MELS + 2] - hz_points[:N_MELS], 1e-12)
    filters *= enorm[:, np.newaxis]
    return filters


def log_mel_spectrogram(waveform: np.ndarray, filters: np.ndarray) -> np.ndarray:
    _, _, stft = signal.stft(
        waveform,
        fs=TARGET_SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft) ** 2
    mel_power = filters @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    log_mel -= np.max(log_mel)
    return np.maximum(log_mel, -80.0)


def choose_sanity_segments(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """각 split에서 human과 12개 생성기 구간을 하나씩 고른다."""
    position_order = {"middle": 0, "beginning": 1, "end": 2}
    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        candidates[(row["split"], row["generator"])].append(row)

    selected = []
    for key in sorted(candidates):
        options = sorted(
            candidates[key],
            key=lambda row: (
                position_order.get(row["segment_position"], 9),
                row["segment_id"],
            ),
        )
        selected.append(options[0])
    return selected


def choose_matched_examples(
    rows: list[dict[str, str]],
) -> tuple[str, str, list[dict[str, str]]]:
    """같은 train 원곡에 속한 REAL·MusicGen·Suno 앞부분을 고른다."""
    required = {"human", "musicgen", "suno"}
    sources_by_group: dict[str, set[str]] = defaultdict(set)
    original_by_group: dict[str, str] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        sources_by_group[row["group_id"]].add(row["generator"])
        original_by_group[row["group_id"]] = row["original_audio"]

    matching_groups = sorted(
        group_id
        for group_id, sources in sources_by_group.items()
        if required.issubset(sources)
    )
    if not matching_groups:
        raise RuntimeError("REAL·MusicGen·Suno가 함께 있는 train 원곡 그룹이 없습니다.")
    group_id = matching_groups[0]

    selected = []
    for generator in ["human", "musicgen", "suno"]:
        options = sorted(
            (
                row
                for row in rows
                if row["group_id"] == group_id
                and row["generator"] == generator
                and row["segment_position"] == "beginning"
            ),
            key=lambda row: row["segment_id"],
        )
        if not options:
            raise RuntimeError(f"{group_id}에 {generator} beginning 구간이 없습니다.")
        selected.append(options[0])
    return group_id, original_by_group[group_id], selected


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "(결과 없음)"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.segment_manifest)
    if len(rows) != 9784:
        raise RuntimeError(f"세그먼트 9,784개를 예상했지만 {len(rows):,}개입니다.")

    # 3개 split × (human + 12개 생성기) = 39개를 실제 디코딩한다.
    sanity_segments = choose_sanity_segments(rows)
    expected_profiles = {
        (row["split"], row["generator"])
        for row in rows
    }
    selected_profiles = {
        (row["split"], row["generator"])
        for row in sanity_segments
    }
    if selected_profiles != expected_profiles:
        missing = sorted(expected_profiles - selected_profiles)
        raise RuntimeError(f"검사용 프로필이 빠졌습니다: {missing}")

    sanity_rows: list[dict[str, object]] = []
    for row in sanity_segments:
        error = ""
        decoded_samples = 0
        padding_samples = 0
        output_samples = 0
        finite = False
        peak = math.nan
        rms = math.nan
        zero_fraction = math.nan
        try:
            waveform, decoded_samples, padding_samples = decode_segment(
                args.ffmpeg,
                row["absolute_path"],
                float(row["start_sec"]),
                float(row["duration_sec"]),
            )
            output_samples = len(waveform)
            finite = bool(np.isfinite(waveform).all())
            peak = float(np.max(np.abs(waveform)))
            rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
            zero_fraction = float(np.mean(np.abs(waveform) < 1e-7))
        except Exception as exc:  # 실패도 CSV에 남긴 뒤 전체 검사를 실패시킨다.
            error = str(exc)
        sanity_rows.append(
            {
                "split": row["split"],
                "label": row["label"],
                "generator": row["generator"],
                "segment_id": row["segment_id"],
                "format_name": row["format_name"],
                "original_sample_rate": row["original_sample_rate"],
                "original_channels": row["original_channels"],
                "decoded_samples": decoded_samples,
                "padding_samples": padding_samples,
                "output_samples": output_samples,
                "output_sample_rate": TARGET_SAMPLE_RATE,
                "output_channels": 1,
                "finite": finite,
                "peak": round(peak, 8) if math.isfinite(peak) else "",
                "rms": round(rms, 8) if math.isfinite(rms) else "",
                "near_zero_fraction": round(zero_fraction, 8)
                if math.isfinite(zero_fraction)
                else "",
                "error": error,
            }
        )
    write_csv(args.output_dir / "audio_sanity_checks.csv", sanity_rows)

    decode_failures = sum(bool(row["error"]) for row in sanity_rows)
    wrong_output_length = sum(
        int(row["output_samples"]) != TARGET_SAMPLES for row in sanity_rows
    )
    non_finite = sum(not bool(row["finite"]) for row in sanity_rows)
    silent = sum(float(row["rms"] or 0.0) < 1e-5 for row in sanity_rows)
    total_padding = sum(int(row["padding_samples"]) for row in sanity_rows)
    quality_rows = [
        {"check": "tested_profiles", "value": len(sanity_rows), "passed": len(sanity_rows) == 39},
        {"check": "decode_failures", "value": decode_failures, "passed": decode_failures == 0},
        {"check": "wrong_output_length", "value": wrong_output_length, "passed": wrong_output_length == 0},
        {"check": "non_finite_waveforms", "value": non_finite, "passed": non_finite == 0},
        {"check": "silent_waveforms", "value": silent, "passed": silent == 0},
        {"check": "total_padding_samples", "value": total_padding, "passed": total_padding == 0},
    ]
    write_csv(args.output_dir / "audio_pipeline_quality.csv", quality_rows)
    if not all(bool(row["passed"]) for row in quality_rows):
        failures = [row for row in quality_rows if not row["passed"]]
        raise RuntimeError(f"오디오 입력 검사 실패: {failures}")

    # 같은 원곡 그룹의 REAL, MusicGen, Suno를 눈과 귀로 비교할 수 있게 저장한다.
    group_id, original_audio, examples = choose_matched_examples(rows)
    filters = mel_filterbank()
    decoded_examples: list[tuple[dict[str, str], np.ndarray, np.ndarray, Path]] = []
    preview_rows: list[dict[str, object]] = []
    for index, row in enumerate(examples, start=1):
        waveform, decoded_samples, padding_samples = decode_segment(
            args.ffmpeg,
            row["absolute_path"],
            float(row["start_sec"]),
            float(row["duration_sec"]),
        )
        spectrogram = log_mel_spectrogram(waveform, filters)
        preview_name = f"{index:02d}_{row['label']}_{row['generator']}.wav"
        preview_path = args.output_dir / "previews" / preview_name
        write_preview_wav(preview_path, waveform)
        decoded_examples.append((row, waveform, spectrogram, preview_path))
        preview_rows.append(
            {
                "group_id": group_id,
                "original_audio": original_audio,
                "label": row["label"],
                "generator": row["generator"],
                "segment_id": row["segment_id"],
                "start_sec": row["start_sec"],
                "decoded_samples": decoded_samples,
                "padding_samples": padding_samples,
                "preview_path": str(preview_path),
            }
        )
    write_csv(args.output_dir / "matched_examples.csv", preview_rows)

    figure, axes = plt.subplots(
        nrows=len(decoded_examples),
        ncols=2,
        figsize=(13, 9),
        gridspec_kw={"width_ratios": [1, 1.45]},
    )
    time_axis = np.arange(TARGET_SAMPLES) / TARGET_SAMPLE_RATE
    image = None
    for row_index, (row, waveform, spectrogram, _) in enumerate(decoded_examples):
        title_name = "Human REAL" if row["label"] == "REAL" else f"FAKE - {row['generator']}"
        axes[row_index, 0].plot(time_axis, waveform, linewidth=0.35, color="#4C78A8")
        axes[row_index, 0].set(
            title=f"{title_name}: waveform",
            xlim=(0, TARGET_DURATION_SEC),
            xlabel="Time (seconds)",
            ylabel="Amplitude",
        )
        image = axes[row_index, 1].imshow(
            spectrogram,
            origin="lower",
            aspect="auto",
            extent=[0, TARGET_DURATION_SEC, 0, N_MELS],
            vmin=-80,
            vmax=0,
            cmap="magma",
        )
        axes[row_index, 1].set(
            title=f"{title_name}: Log-Mel",
            xlabel="Time (seconds)",
            ylabel="Mel band",
        )
    figure.suptitle(
        "Matched original group: REAL vs fully AI-generated music",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.87,
        bottom=0.06,
        top=0.94,
        hspace=0.42,
        wspace=0.22,
    )
    if image is not None:
        colorbar_axis = figure.add_axes([0.9, 0.16, 0.015, 0.68])
        figure.colorbar(image, cax=colorbar_axis, label="Relative power (dB)")
    figure.savefig(
        args.output_dir / "matched_waveform_logmel.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)

    report = f"""# 3단계 오디오 입력 및 스펙트로그램 결과

## 이 단계에서 한 일

앞 단계의 9,784개 구간 중 Train/Validation/Test 각각에서 인간 음악과
12개 AI 생성기 음악을 하나씩 뽑아 총 39개를 실제로 디코딩했다. MP3와 WAV,
16kHz부터 48kHz까지 서로 다른 원본이 모두 같은 모델 입력으로 변환되는지
검사했다.

## 모델에 들어가는 공통 모양

- 길이: 10초
- 채널: mono 1채널
- 샘플링레이트: 24,000Hz
- 숫자 개수: 240,000개
- 파형 자료형: float32

오디오의 원래 형식이 달라도 모델에는 항상 같은 모양의 숫자 배열이 들어간다.

## 입력 파이프라인 품질 검사

{markdown_table(quality_rows)}

39개 대표 프로필이 모두 디코딩되었고, 길이 부족으로 0을 덧붙인 구간도 없다.
앞 단계에서 전체 9,784개 구간의 파일 존재·디코딩 가능 여부·시작과 끝 경계를
검사했으므로 이 입력 방식을 전체 데이터에 적용할 수 있다.

## 같은 원곡 계열 비교

- 그룹: `{group_id}`
- 원곡 표기: `{original_audio}`
- 비교 대상: Human REAL, MusicGen FAKE, Suno FAKE
- 그림: `matched_waveform_logmel.png`
- 직접 들을 파일: `previews` 폴더의 WAV 3개

파형은 시간에 따른 소리의 세기를 보여주고, Log-Mel 스펙트로그램은 시간에
따라 저음부터 고음까지 에너지가 어떻게 분포하는지 색으로 보여준다. 그림에서
차이가 보여도 그것만으로 AI 생성의 일반적 특징이라고 결론 내리면 안 된다.
모델 학습과 전체 테스트 결과로 확인해야 한다.

## 다음 단계

각 10초 구간에서 MFCC와 스펙트럴 특징을 숫자로 추출하고, 첫 기계학습
기준선인 Logistic Regression과 SVM을 학습한다. Train으로 공부하고,
Validation으로 설정을 고른 뒤, Test는 마지막 평가에만 사용한다.
"""
    (args.output_dir / "AUDIO_EDA_결과요약.md").write_text(report, encoding="utf-8")

    print("오디오 입력 및 Log-Mel 검사가 완료되었습니다.")
    print(f"실제 디코딩 검사: {len(sanity_rows)}개 프로필")
    print(f"공통 입력: mono, {TARGET_SAMPLE_RATE:,}Hz, {TARGET_DURATION_SEC:.0f}초, {TARGET_SAMPLES:,} samples")
    print(f"품질 검사 통과: {sum(bool(row['passed']) for row in quality_rows)}/{len(quality_rows)}")
    print(f"비교 그룹: {original_audio}")
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
