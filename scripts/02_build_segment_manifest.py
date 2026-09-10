#!/usr/bin/env python3
"""2단계: 각 트랙에서 읽을 최대 3개의 10초 구간 위치를 만든다.

원본 오디오를 잘라서 새 파일로 저장하지 않는다. 학습 코드가 나중에
원본 파일의 start_sec 위치부터 10초만 읽을 수 있도록 CSV에 위치만 적는다.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT_DURATION_SEC = 10.0
NEAR_30_SEC_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_manifest.csv",
    )
    parser.add_argument(
        "--audio-validation",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_audio_validation.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_summary.csv",
    )
    parser.add_argument(
        "--generator-summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_generator_summary.csv",
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_quality_checks.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "SEGMENT_결과요약.md",
    )
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


def normalize_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def select_windows(duration_sec: float) -> list[tuple[str, float]]:
    """트랙 길이에 따라 (위치 이름, 시작 초) 목록을 반환한다."""
    if duration_sec + NEAR_30_SEC_TOLERANCE >= 30.0:
        # 약 30초인 FMA는 인코딩 과정에서 수 ms 짧아질 수 있다. 실제 길이
        # 안에서 앞/중간/끝 10초를 고르면 겹침은 최대 tolerance의 절반이다.
        return [
            ("beginning", 0.0),
            ("middle", max(0.0, (duration_sec - SEGMENT_DURATION_SEC) / 2)),
            ("end", max(0.0, duration_sec - SEGMENT_DURATION_SEC)),
        ]
    if duration_sec >= 20.0:
        return [
            ("beginning", 0.0),
            ("end", duration_sec - SEGMENT_DURATION_SEC),
        ]
    if duration_sec >= SEGMENT_DURATION_SEC:
        return [("middle", (duration_sec - SEGMENT_DURATION_SEC) / 2)]
    return []


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
    tracks = read_csv(args.track_manifest)
    validations = read_csv(args.audio_validation)

    validation_by_sample = {row["sample_id"]: row for row in validations}
    if len(validation_by_sample) != len(validations):
        raise RuntimeError("오디오 검증표에 중복 sample_id가 있습니다.")

    segment_rows: list[dict[str, object]] = []
    track_segment_counts: Counter[str] = Counter()
    segment_count_distribution: Counter[int] = Counter()

    for track in tracks:
        sample_id = track["sample_id"]
        validation = validation_by_sample.get(sample_id)
        if validation is None:
            raise RuntimeError(f"오디오 검증 결과가 없습니다: {sample_id}")
        if not normalize_bool(validation["exists"]):
            raise RuntimeError(f"오디오 파일이 없습니다: {sample_id}")
        if not normalize_bool(validation["decodable"]):
            raise RuntimeError(f"디코딩할 수 없는 오디오입니다: {sample_id}")
        if not normalize_bool(validation["usable_10s"]):
            raise RuntimeError(f"10초보다 짧은 오디오입니다: {sample_id}")

        track_duration = float(validation["actual_duration_sec"])
        windows = select_windows(track_duration)
        if not windows:
            raise RuntimeError(f"선택 가능한 10초 구간이 없습니다: {sample_id}")
        segment_count_distribution[len(windows)] += 1

        for segment_index, (position, start_sec) in enumerate(windows, start=1):
            end_sec = start_sec + SEGMENT_DURATION_SEC
            if start_sec < -1e-9 or end_sec > track_duration + 1e-6:
                raise RuntimeError(
                    f"구간이 트랙 경계를 벗어났습니다: {sample_id}, "
                    f"start={start_sec}, end={end_sec}, track={track_duration}"
                )
            segment_rows.append(
                {
                    "segment_id": f"{sample_id}_seg{segment_index:02d}",
                    "sample_id": sample_id,
                    "group_id": track["group_id"],
                    "original_audio": track["original_audio"],
                    "absolute_path": track["absolute_path"],
                    "label": track["label"],
                    "label_id": track["label_id"],
                    "source": track["source"],
                    "generator": track["generator"],
                    "generation_type": track["generation_type"],
                    "genre": track["genre"],
                    "split": track["split"],
                    "format_name": validation["format_name"],
                    "codec_name": validation["codec_name"],
                    "original_sample_rate": validation["sample_rate"],
                    "original_channels": validation["channels"],
                    "track_duration_sec": round(track_duration, 6),
                    "segments_in_track": len(windows),
                    "segment_index": segment_index,
                    "segment_position": position,
                    "start_sec": round(start_sec, 6),
                    "duration_sec": SEGMENT_DURATION_SEC,
                    "end_sec": round(end_sec, 6),
                }
            )
            track_segment_counts[sample_id] += 1

    write_csv(args.output, segment_rows)

    # split과 label별 요약
    summary_accumulator: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"segments": 0, "sample_ids": set(), "group_ids": set()}
    )
    generator_accumulator: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"segments": 0, "sample_ids": set(), "group_ids": set()}
    )
    for row in segment_rows:
        split_key = (str(row["split"]), str(row["label"]))
        split_bucket = summary_accumulator[split_key]
        split_bucket["segments"] = int(split_bucket["segments"]) + 1
        split_bucket["sample_ids"].add(row["sample_id"])
        split_bucket["group_ids"].add(row["group_id"])

        if row["label"] == "FAKE":
            generator_key = (str(row["split"]), str(row["generator"]))
            generator_bucket = generator_accumulator[generator_key]
            generator_bucket["segments"] = int(generator_bucket["segments"]) + 1
            generator_bucket["sample_ids"].add(row["sample_id"])
            generator_bucket["group_ids"].add(row["group_id"])

    split_order = {"train": 0, "validation": 1, "test": 2}
    label_order = {"REAL": 0, "FAKE": 1}
    summary_rows: list[dict[str, object]] = []
    for (split, label), values in sorted(
        summary_accumulator.items(),
        key=lambda item: (split_order[item[0][0]], label_order[item[0][1]]),
    ):
        segments = int(values["segments"])
        summary_rows.append(
            {
                "split": split,
                "label": label,
                "tracks": len(values["sample_ids"]),
                "unique_groups": len(values["group_ids"]),
                "segments": segments,
                "segment_hours": round(segments * SEGMENT_DURATION_SEC / 3600, 4),
            }
        )
    write_csv(args.summary_output, summary_rows)

    generator_rows: list[dict[str, object]] = []
    for (split, generator), values in sorted(
        generator_accumulator.items(),
        key=lambda item: (split_order[item[0][0]], item[0][1]),
    ):
        generator_rows.append(
            {
                "split": split,
                "generator": generator,
                "tracks": len(values["sample_ids"]),
                "unique_groups": len(values["group_ids"]),
                "segments": int(values["segments"]),
            }
        )
    write_csv(args.generator_summary_output, generator_rows)

    # 품질 검증: 경계, 중복, split 누수, 트랙당 구간 수를 다시 검사한다.
    segment_ids = [str(row["segment_id"]) for row in segment_rows]
    group_splits: dict[str, set[str]] = defaultdict(set)
    windows_by_sample: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in segment_rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
        windows_by_sample[str(row["sample_id"])].append(
            (float(row["start_sec"]), float(row["end_sec"]))
        )

    max_overlap = 0.0
    for windows in windows_by_sample.values():
        ordered = sorted(windows)
        for previous, current in zip(ordered, ordered[1:]):
            max_overlap = max(max_overlap, previous[1] - current[0])

    boundary_errors = sum(
        float(row["start_sec"]) < -1e-9
        or float(row["end_sec"]) > float(row["track_duration_sec"]) + 1e-6
        for row in segment_rows
    )
    leaking_groups = sum(len(splits) > 1 for splits in group_splits.values())
    wrong_track_counts = sum(
        count < 1 or count > 3 for count in track_segment_counts.values()
    )

    quality_rows = [
        {"check": "total_segments", "value": len(segment_rows), "passed": len(segment_rows) == 9784},
        {"check": "duplicate_segment_ids", "value": len(segment_ids) - len(set(segment_ids)), "passed": len(segment_ids) == len(set(segment_ids))},
        {"check": "tracks_without_segments", "value": len(tracks) - len(track_segment_counts), "passed": len(tracks) == len(track_segment_counts)},
        {"check": "tracks_outside_1_to_3_segments", "value": wrong_track_counts, "passed": wrong_track_counts == 0},
        {"check": "segment_boundary_errors", "value": boundary_errors, "passed": boundary_errors == 0},
        {"check": "groups_in_multiple_splits", "value": leaking_groups, "passed": leaking_groups == 0},
        {"check": "max_overlap_seconds", "value": round(max_overlap, 6), "passed": max_overlap <= NEAR_30_SEC_TOLERANCE / 2 + 1e-6},
    ]
    write_csv(args.quality_output, quality_rows)

    if not all(bool(row["passed"]) for row in quality_rows):
        failures = [row for row in quality_rows if not row["passed"]]
        raise RuntimeError(f"세그먼트 품질 검사 실패: {failures}")

    real_segments = sum(row["segments"] for row in summary_rows if row["label"] == "REAL")
    fake_segments = sum(row["segments"] for row in summary_rows if row["label"] == "FAKE")
    distribution_rows = [
        {"트랙당_구간수": count, "트랙수": segment_count_distribution[count]}
        for count in sorted(segment_count_distribution)
    ]
    report = f"""# 2단계 10초 세그먼트 구성 결과

## 이 단계에서 한 일

원본 음악 파일을 새로 자르거나 복사하지 않고, 각 트랙에서 읽을 10초 구간의
시작 위치를 `segment_manifest.csv`에 기록했다. 따라서 추가 저장공간은 거의
사용하지 않으며 원본 파일도 변경하지 않았다.

## 구간 선택 규칙

- 30초 이상: 앞, 중간, 끝에서 3개
- 20초 이상 30초 미만: 앞, 끝에서 2개
- 10초 이상 20초 미만: 가운데에서 1개
- 약 30초인 FMA는 MP3 인코딩 오차 0.02초를 허용하여 3개로 처리

구간 길이 계산에는 manifest의 반올림 값이 아니라 `ffprobe`가 측정한 실제
오디오 길이를 사용했다.

## 생성 결과

{markdown_table(summary_rows)}

- 전체: {len(segment_rows):,}개
- REAL: {real_segments:,}개
- FAKE: {fake_segments:,}개
- 10초 구간을 모두 이어 들었을 때: {len(segment_rows) * 10 / 3600:.2f}시간

## 트랙 길이에 따른 구간 수

{markdown_table(distribution_rows)}

## 품질 검사

{markdown_table(quality_rows)}

약 30초인 FMA 일부는 실제 길이가 수 ms 짧아 앞·중간·끝 구간 사이에 최대
{max_overlap:.6f}초의 미세한 겹침이 있다. 이는 허용치 0.01초보다 작다.

## CSV 한 행의 의미

예를 들어 `start_sec=20`, `duration_sec=10`이면 학습할 때 해당 음악 파일의
20초 지점부터 30초 지점까지 읽는다는 뜻이다. `label`은 REAL 또는 FAKE이고,
`split`은 공부용(train), 모의고사용(validation), 최종 시험용(test)을 뜻한다.

## 다음 단계

REAL과 FAKE 대표 구간을 실제로 읽어 모두 mono·24kHz로 변환할 수 있는지
검사하고, Log-Mel 스펙트로그램을 그려 눈으로 비교한다.
"""
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")

    print("10초 세그먼트 manifest 생성이 완료되었습니다.")
    print(f"전체: {len(segment_rows):,}개 (REAL {real_segments:,}, FAKE {fake_segments:,})")
    print(
        "트랙당 구간 수: "
        + ", ".join(
            f"{count}개={segment_count_distribution[count]:,}곡"
            for count in sorted(segment_count_distribution)
        )
    )
    print(f"품질 검사 통과: {sum(bool(row['passed']) for row in quality_rows)}/{len(quality_rows)}")
    print(f"결과: {args.output}")


if __name__ == "__main__":
    main()
