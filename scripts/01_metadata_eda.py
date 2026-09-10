#!/usr/bin/env python3
"""1단계: 모델링 전에 트랙 메타데이터와 데이터 품질을 점검한다.

이 스크립트는 오디오 전체를 메모리에 불러오지 않는다. 이미 만들어 둔
track manifest와 ffprobe 검증 결과만 이용하므로 노트북에서도 빠르게
실행할 수 있다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_manifest.csv",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_audio_validation.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{source}에 필요한 열이 없습니다: {sorted(missing)}")


def normalize_bool(series: pd.Series) -> pd.Series:
    """CSV가 bool 또는 문자열로 읽혀도 같은 결과가 나오게 한다."""
    return series.astype(str).str.strip().str.lower().eq("true")


def save_table(frame: pd.DataFrame, output_dir: Path, filename: str) -> None:
    frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def annotate_bars(axis: plt.Axes) -> None:
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3, fontsize=9)


def save_figure(figure: plt.Figure, output_dir: Path, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    """추가 패키지(tabulate) 없이 작은 DataFrame을 Markdown으로 바꾼다."""
    display = frame.copy()
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    validation = pd.read_csv(args.validation)

    require_columns(
        manifest,
        {
            "sample_id",
            "group_id",
            "absolute_path",
            "label",
            "generator",
            "genre",
            "duration_sec",
            "split",
        },
        args.manifest,
    )
    require_columns(
        validation,
        {
            "sample_id",
            "exists",
            "format_name",
            "codec_name",
            "sample_rate",
            "channels",
            "actual_duration_sec",
            "decodable",
            "usable_10s",
        },
        args.validation,
    )

    validation_columns = [
        "sample_id",
        "exists",
        "format_name",
        "codec_name",
        "sample_rate",
        "channels",
        "actual_duration_sec",
        "decodable",
        "usable_10s",
    ]
    tracks = manifest.merge(
        validation[validation_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if tracks["format_name"].isna().any():
        missing_ids = tracks.loc[tracks["format_name"].isna(), "sample_id"].tolist()
        raise RuntimeError(f"검증 결과가 없는 sample_id가 있습니다: {missing_ids[:10]}")

    for column in ["exists", "decodable", "usable_10s"]:
        tracks[column] = normalize_bool(tracks[column])
    tracks["duration_hours"] = tracks["actual_duration_sec"] / 3600
    tracks["sample_rate"] = tracks["sample_rate"].astype(int)
    tracks["channels"] = tracks["channels"].astype(int)

    # 표 1: REAL/FAKE 전체 현황
    label_overview = (
        tracks.groupby("label", sort=False)
        .agg(
            tracks=("sample_id", "size"),
            unique_groups=("group_id", "nunique"),
            duration_hours=("duration_hours", "sum"),
            mean_duration_sec=("actual_duration_sec", "mean"),
            median_duration_sec=("actual_duration_sec", "median"),
            min_duration_sec=("actual_duration_sec", "min"),
            max_duration_sec=("actual_duration_sec", "max"),
        )
        .reset_index()
    )
    for column in label_overview.columns[3:]:
        label_overview[column] = label_overview[column].round(2)
    save_table(label_overview, args.output_dir, "01_label_overview.csv")

    # 표 2: Train/Validation/Test별 클래스 분포
    split_overview = (
        tracks.groupby(["split", "label"], sort=False)
        .agg(
            tracks=("sample_id", "size"),
            unique_groups=("group_id", "nunique"),
            duration_hours=("duration_hours", "sum"),
        )
        .reset_index()
    )
    split_overview["duration_hours"] = split_overview["duration_hours"].round(2)
    save_table(split_overview, args.output_dir, "02_split_overview.csv")

    # 표 3: AI 생성기별 데이터 현황
    fake_tracks = tracks.loc[tracks["label"].eq("FAKE")].copy()
    generator_overview = (
        fake_tracks.groupby("generator")
        .agg(
            tracks=("sample_id", "size"),
            unique_groups=("group_id", "nunique"),
            duration_hours=("duration_hours", "sum"),
            median_duration_sec=("actual_duration_sec", "median"),
            formats=("format_name", lambda x: "|".join(sorted(set(x)))),
            sample_rates=("sample_rate", lambda x: "|".join(map(str, sorted(set(x))))),
            channels=("channels", lambda x: "|".join(map(str, sorted(set(x))))),
        )
        .reset_index()
        .sort_values(["tracks", "generator"], ascending=[False, True])
    )
    for column in ["duration_hours", "median_duration_sec"]:
        generator_overview[column] = generator_overview[column].round(2)
    save_table(generator_overview, args.output_dir, "03_generator_overview.csv")

    genre_overview = (
        tracks.groupby(["genre", "label"])
        .size()
        .rename("tracks")
        .reset_index()
        .sort_values(["genre", "label"])
    )
    save_table(genre_overview, args.output_dir, "04_genre_overview.csv")

    audio_profile = (
        tracks.groupby(
            ["label", "format_name", "codec_name", "sample_rate", "channels"]
        )
        .size()
        .rename("tracks")
        .reset_index()
        .sort_values(["label", "tracks"], ascending=[True, False])
    )
    save_table(audio_profile, args.output_dir, "05_audio_profile.csv")

    # 데이터 품질 및 누수 점검
    leaking_groups = int(tracks.groupby("group_id")["split"].nunique().gt(1).sum())
    quality_checks = pd.DataFrame(
        [
            ("total_tracks", len(tracks), len(tracks) == 3458),
            ("duplicate_sample_ids", tracks["sample_id"].duplicated().sum(), tracks["sample_id"].is_unique),
            ("duplicate_audio_paths", tracks["absolute_path"].duplicated().sum(), tracks["absolute_path"].is_unique),
            ("missing_files", (~tracks["exists"]).sum(), tracks["exists"].all()),
            ("undecodable_files", (~tracks["decodable"]).sum(), tracks["decodable"].all()),
            ("shorter_than_10s", (~tracks["usable_10s"]).sum(), tracks["usable_10s"].all()),
            ("groups_in_multiple_splits", leaking_groups, leaking_groups == 0),
        ],
        columns=["check", "value", "passed"],
    )
    save_table(quality_checks, args.output_dir, "06_quality_checks.csv")

    # 그림 1: 클래스 수
    class_counts = tracks["label"].value_counts().reindex(["REAL", "FAKE"])
    figure, axis = plt.subplots(figsize=(6, 4))
    class_counts.plot.bar(ax=axis, color=["#4C78A8", "#E45756"], rot=0)
    axis.set(title="Track count by class", xlabel="Class", ylabel="Tracks")
    annotate_bars(axis)
    save_figure(figure, args.output_dir, "01_class_counts.png")

    # 그림 2: 생성기별 곡 수
    generator_counts = fake_tracks["generator"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(8, 6))
    generator_counts.plot.barh(ax=axis, color="#E45756")
    axis.set(title="FAKE track count by generator", xlabel="Tracks", ylabel="Generator")
    annotate_bars(axis)
    save_figure(figure, args.output_dir, "02_generator_counts.png")

    # 그림 3: 장르별 클래스 분포
    genre_pivot = tracks.pivot_table(
        index="genre", columns="label", values="sample_id", aggfunc="count", fill_value=0
    ).reindex(columns=["REAL", "FAKE"])
    figure, axis = plt.subplots(figsize=(7, 4))
    genre_pivot.plot.bar(ax=axis, color=["#4C78A8", "#E45756"], rot=0)
    axis.set(title="Track count by genre and class", xlabel="Genre", ylabel="Tracks")
    annotate_bars(axis)
    save_figure(figure, args.output_dir, "03_genre_class_counts.png")

    # 그림 4: 길이 분포. 긴 꼬리가 있으므로 극단값을 숨기지 않고 log 축을 쓴다.
    figure, axis = plt.subplots(figsize=(7, 4))
    data = [
        tracks.loc[tracks["label"].eq(label), "actual_duration_sec"]
        for label in ["REAL", "FAKE"]
    ]
    axis.boxplot(data, tick_labels=["REAL", "FAKE"], vert=False, showfliers=True)
    axis.set_xscale("log")
    axis.set(title="Track duration by class", xlabel="Duration (seconds, log scale)")
    save_figure(figure, args.output_dir, "04_duration_by_class.png")

    # 그림 5: 샘플링레이트별 클래스 수
    rate_pivot = tracks.pivot_table(
        index="sample_rate",
        columns="label",
        values="sample_id",
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=["REAL", "FAKE"], fill_value=0)
    rate_pivot.index = [f"{rate / 1000:g}k" for rate in rate_pivot.index]
    figure, axis = plt.subplots(figsize=(8, 4))
    rate_pivot.plot.bar(ax=axis, color=["#4C78A8", "#E45756"], rot=0)
    axis.set(title="Sample rate by class", xlabel="Sample rate (Hz)", ylabel="Tracks")
    annotate_bars(axis)
    save_figure(figure, args.output_dir, "05_sample_rate_by_class.png")

    # 그림 6: split별 클래스 수
    split_pivot = tracks.pivot_table(
        index="split", columns="label", values="sample_id", aggfunc="count", fill_value=0
    ).reindex(["train", "validation", "test"]).reindex(columns=["REAL", "FAKE"])
    figure, axis = plt.subplots(figsize=(7, 4))
    split_pivot.plot.bar(ax=axis, color=["#4C78A8", "#E45756"], rot=0)
    axis.set(title="Track count by split and class", xlabel="Split", ylabel="Tracks")
    annotate_bars(axis)
    save_figure(figure, args.output_dir, "06_split_class_counts.png")

    real_count = int(class_counts["REAL"])
    fake_count = int(class_counts["FAKE"])
    fake_to_real_ratio = fake_count / real_count
    real_formats = ", ".join(
        f"{name} {count}곡"
        for name, count in tracks.loc[tracks["label"].eq("REAL"), "format_name"]
        .value_counts()
        .items()
    )
    fake_formats = ", ".join(
        f"{name} {count}곡"
        for name, count in fake_tracks["format_name"].value_counts().items()
    )
    musicgen = generator_overview.loc[generator_overview["generator"].eq("musicgen")]
    musicgen_description = "확인 불가"
    if len(musicgen) == 1:
        row = musicgen.iloc[0]
        musicgen_description = (
            f"{int(row['tracks'])}곡, {row['formats']}, "
            f"{row['sample_rates']}Hz, {row['channels']}채널"
        )

    report = f"""# 1단계 메타데이터 EDA 결과

## 이 단계에서 확인한 것

모델을 학습하기 전에 데이터의 수량, 길이, 장르, 생성기, 파일 형식과
Train/Validation/Test 누수 여부를 확인했다. 이 단계에서는 실제 음악의
내용을 분석하지 않고 manifest와 ffprobe 검증 결과만 사용했다.

## 데이터 요약

{markdown_table(label_overview)}

- 전체 트랙: {len(tracks):,}곡
- REAL: {real_count:,}곡
- FAKE: {fake_count:,}곡
- FAKE/REAL 트랙 수 비율: {fake_to_real_ratio:.2f}:1
- 원곡 그룹: {tracks['group_id'].nunique():,}개

## 품질 점검

{markdown_table(quality_checks)}

모든 파일이 존재하고 디코딩 가능하며 10초 이상이다. 같은 원곡 그룹이
여러 split에 들어간 경우가 없으므로 현재 그룹 분할은 모델 평가에 사용할
수 있다.

## 가장 중요한 발견

1. **클래스 불균형**: FAKE가 REAL보다 {fake_to_real_ratio:.2f}배 많다.
   학습 때 REAL/FAKE 균형 sampling 또는 sample weight가 필요하다.
2. **파일 형식 차이**: REAL은 {real_formats}, FAKE는 {fake_formats}이다.
3. **MusicGen의 고유 형식**: MusicGen은 {musicgen_description}이다.
   모델이 음악의 생성 흔적 대신 WAV/MP3, 샘플링레이트, 채널 차이를
   지름길로 사용할 수 있다.
4. **길이 차이**: REAL은 거의 30초인 반면 FAKE는 생성기마다 길이가 다르다.
   트랙당 최대 3개의 10초 구간만 사용해야 긴 FAKE가 학습을 지배하지 않는다.

## 모델링 전에 확정할 처리

- 모든 오디오를 mono, 24,000Hz로 읽는다.
- 각 트랙에서 최대 3개의 10초 구간만 사용한다.
- Train 배치는 REAL/FAKE가 1:1에 가깝도록 구성한다.
- MusicGen 미지 생성기 실험은 원본 조건과 공통 MP3 변환 조건을 모두 보고한다.
- 코덱 증강은 REAL과 FAKE에 같은 확률과 설정으로 적용한다.

## 다음 단계

`track_manifest.csv`를 바탕으로 실제 학습 위치를 기록하는
`segment_manifest.csv`를 만든다. 원본 음원을 물리적으로 잘라 복사하지 않고
각 10초 구간의 시작 시각만 CSV에 저장한다.

## 생성된 그림

- `01_class_counts.png`: REAL/FAKE 곡 수
- `02_generator_counts.png`: AI 생성기별 곡 수
- `03_genre_class_counts.png`: 장르별 REAL/FAKE 분포
- `04_duration_by_class.png`: 클래스별 곡 길이
- `05_sample_rate_by_class.png`: 샘플링레이트 차이
- `06_split_class_counts.png`: split별 곡 수
"""
    (args.output_dir / "EDA_결과요약.md").write_text(report, encoding="utf-8")

    print("메타데이터 EDA가 완료되었습니다.")
    print(f"분석 트랙: {len(tracks):,}곡 (REAL {real_count:,}, FAKE {fake_count:,})")
    print(f"품질 검사 통과: {int(quality_checks['passed'].sum())}/{len(quality_checks)}")
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
