#!/usr/bin/env python3
"""외부 음원을 공통 MP3 조건의 ML·CNN 모델로 곡 단위 판정한다."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="판정할 외부 오디오 파일")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "external_predictions",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audio_profile(ffprobe: str, audio_path: Path) -> dict[str, object]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name,size,bit_rate:stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    streams = payload.get("streams", [])
    audio_stream = streams[0] if streams else {}
    return {
        "duration_sec": duration,
        "format_name": payload["format"].get("format_name"),
        "size_bytes": int(payload["format"].get("size", 0)),
        "bit_rate": int(payload["format"].get("bit_rate", 0)),
        "codec_name": audio_stream.get("codec_name"),
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
    }


def segment_starts(duration: float) -> list[float]:
    if duration < 10.0:
        raise ValueError(f"오디오가 10초보다 짧습니다: {duration:.2f}초")
    if duration >= 30.0:
        candidates = [0.0, (duration - 10.0) / 2.0, duration - 10.0]
    elif duration >= 20.0:
        candidates = [0.0, duration - 10.0]
    else:
        candidates = [(duration - 10.0) / 2.0]
    return list(dict.fromkeys(round(value, 6) for value in candidates))


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS를 사용할 수 없습니다.")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def continuous_score(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def safe_stem(path: Path) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", path.stem).strip("_")
    return cleaned or "external_audio"


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    feature_module = load_script_module(
        "echoes_handcrafted_features",
        PROJECT_ROOT / "scripts" / "04_extract_handcrafted_features.py",
    )
    logmel_module = load_script_module(
        "echoes_logmel_features",
        PROJECT_ROOT / "scripts" / "08_extract_logmel_cache.py",
    )
    cnn_module = load_script_module(
        "echoes_logmel_cnn",
        PROJECT_ROOT / "scripts" / "09_train_logmel_cnn.py",
    )

    profile = audio_profile(args.ffprobe, audio_path)
    starts = segment_starts(float(profile["duration_sec"]))
    handcrafted_filterbank = feature_module.make_mel_filterbank()
    cnn_filterbank = logmel_module.mel_filterbank()
    feature_vectors: list[np.ndarray] = []
    logmels: list[np.ndarray] = []
    extracted_names: list[str] | None = None

    for index, start in enumerate(starts, start=1):
        row = {
            "segment_id": f"external_seg{index:02d}",
            "absolute_path": str(audio_path),
            "start_sec": str(start),
            "duration_sec": "10.0",
        }
        waveform = feature_module.decode_segment(args.ffmpeg, row)
        waveform = feature_module.mp3_roundtrip(args.ffmpeg, waveform, 64)
        vector, names = feature_module.extract_feature_vector(
            waveform, handcrafted_filterbank
        )
        if extracted_names is None:
            extracted_names = names
        elif names != extracted_names:
            raise RuntimeError("구간마다 특징 이름 순서가 다릅니다.")
        feature_vectors.append(vector)
        logmels.append(logmel_module.extract_logmel(waveform, cnn_filterbank))

    features = np.stack(feature_vectors).astype(np.float32)
    logmel_array = np.stack(logmels).astype(np.float32)
    training_features = np.load(
        PROJECT_ROOT
        / "artifacts"
        / "features"
        / "handcrafted_features_mp3_64kbps.npz",
        allow_pickle=True,
    )
    expected_names = training_features["feature_names"].astype(str).tolist()
    if extracted_names != expected_names:
        raise RuntimeError("외부 음원과 학습 데이터의 특징 순서가 다릅니다.")

    ml_root = PROJECT_ROOT / "artifacts" / "ml_codec_control_mp3_64kbps"
    ml_metadata = json.loads(
        (ml_root / "training_metadata.json").read_text(encoding="utf-8")
    )
    segment_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    for slug in ["logistic_regression", "rbf_svm"]:
        model = joblib.load(ml_root / "models" / f"{slug}.joblib")
        scores = continuous_score(model, features)
        metadata = ml_metadata["models"][slug]
        segment_threshold = float(metadata["validation_segment_threshold"])
        track_threshold = float(metadata["validation_track_threshold"])
        for index, (start, score) in enumerate(zip(starts, scores), start=1):
            segment_rows.append(
                {
                    "model": metadata["name"],
                    "segment": index,
                    "start_sec": start,
                    "end_sec": start + 10.0,
                    "score": float(score),
                    "threshold": segment_threshold,
                    "prediction": "FAKE" if score >= segment_threshold else "REAL",
                }
            )
        mean_score = float(np.mean(scores))
        track_rows.append(
            {
                "model": metadata["name"],
                "mean_score": mean_score,
                "validation_track_threshold": track_threshold,
                "margin_from_threshold": mean_score - track_threshold,
                "prediction": "FAKE" if mean_score >= track_threshold else "REAL",
            }
        )

    device = choose_device(args.device)
    checkpoint_path = (
        PROJECT_ROOT / "artifacts" / "cnn_mp3_64kbps" / "small_logmel_cnn.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cnn = cnn_module.SmallLogMelCNN().to(device)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()
    cnn_inputs = torch.from_numpy((logmel_array + 40.0) / 40.0).unsqueeze(1).to(device)
    with torch.no_grad():
        cnn_scores = cnn(cnn_inputs).cpu().numpy().astype(np.float64)
    cnn_segment_threshold = float(checkpoint["validation_segment_threshold"])
    cnn_track_threshold = float(checkpoint["validation_track_threshold"])
    for index, (start, score) in enumerate(zip(starts, cnn_scores), start=1):
        segment_rows.append(
            {
                "model": "Log-Mel CNN",
                "segment": index,
                "start_sec": start,
                "end_sec": start + 10.0,
                "score": float(score),
                "threshold": cnn_segment_threshold,
                "prediction": "FAKE" if score >= cnn_segment_threshold else "REAL",
            }
        )
    cnn_mean_score = float(np.mean(cnn_scores))
    track_rows.append(
        {
            "model": "Log-Mel CNN",
            "mean_score": cnn_mean_score,
            "validation_track_threshold": cnn_track_threshold,
            "margin_from_threshold": cnn_mean_score - cnn_track_threshold,
            "prediction": "FAKE" if cnn_mean_score >= cnn_track_threshold else "REAL",
        }
    )

    segment_frame = pd.DataFrame(segment_rows)
    track_frame = pd.DataFrame(track_rows)
    fake_votes = int(track_frame["prediction"].eq("FAKE").sum())
    consensus = "FAKE" if fake_votes >= 2 else "REAL"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(audio_path)
    segment_path = args.output_dir / f"{stem}_segments.csv"
    track_path = args.output_dir / f"{stem}_track.csv"
    report_path = args.output_dir / f"{stem}_결과.md"
    segment_frame.to_csv(segment_path, index=False, encoding="utf-8-sig")
    track_frame.to_csv(track_path, index=False, encoding="utf-8-sig")

    display = track_frame.copy()
    for column in [
        "mean_score",
        "validation_track_threshold",
        "margin_from_threshold",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    report = f"""# 외부 음원 판정 결과

## 입력

- 파일: `{audio_path.name}`
- 길이: {float(profile['duration_sec']):.2f}초
- 원본: {profile['codec_name']}, {profile['sample_rate']}Hz, {profile['channels']}채널
- 분석 위치: {', '.join(f'{value:.2f}초' for value in starts)}부터 각 10초
- 모델 입력: mono·24kHz 및 공통 MP3 64kbps round-trip

## 곡 단위 판정

{display.to_markdown(index=False)}

- 다수결 판정: **{consensus}** ({fake_votes}/3 모델이 FAKE)

`score`는 모델마다 척도가 다른 비보정 점수이므로 서로 크기를 직접 비교하거나
확률로 해석하면 안 된다. 각 모델 내부에서 Validation으로 고정한 threshold보다
높은지만 판정에 사용했다.

이 결과는 외부 한 곡에 대한 사례 분석이다. 이 곡의 실제 생성 출처를 모델에
알려주지 않고 예측했으며, 한 곡만으로 모델 정확도나 일반화 성능을 계산할 수 없다.
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"파일: {audio_path.name}")
    print(f"길이: {float(profile['duration_sec']):.2f}초, 분석 구간: {starts}")
    print(track_frame.to_string(index=False))
    print(f"다수결 판정: {consensus} ({fake_votes}/3 FAKE)")
    print(f"상세 결과: {report_path}")


if __name__ == "__main__":
    main()
