#!/usr/bin/env python3
"""Create a leakage-safe REAL/TTA track manifest and group-level data split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECHOES_ROOT = Path(os.environ.get("ECHOES_ROOT", PROJECT_ROOT / "data" / "Echoes"))


def allocate_counts(total: int, ratios: tuple[float, ...]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def stable_group_id(original_audio: str) -> str:
    digest = hashlib.sha1(original_audio.encode("utf-8")).hexdigest()[:12]
    return f"grp_{digest}"


def stable_fake_id(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"fake_{digest}"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--echoes-root",
        type=Path,
        default=ECHOES_ROOT,
    )
    parser.add_argument(
        "--real-mapping",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "real_track_map.csv",
    )
    parser.add_argument(
        "--real-validation",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "fma_real_validation.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "track_manifest.csv",
    )
    parser.add_argument(
        "--groups-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "split_groups.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "split_summary.csv",
    )
    parser.add_argument(
        "--generator-summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "generator_split_summary.csv",
    )
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "excluded_tta_duplicates.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    echoes_manifest = args.echoes_root / "dataset_manifest.csv"
    raw_echoes_rows = [row for row in load_csv(echoes_manifest) if row["type"] == "TTA"]
    real_rows = load_csv(args.real_mapping)
    validation_rows = load_csv(args.real_validation)

    if len(raw_echoes_rows) != 3165:
        raise RuntimeError(f"Expected 3,165 TTA rows, found {len(raw_echoes_rows)}")
    if len(real_rows) != 296:
        raise RuntimeError(f"Expected 296 REAL rows, found {len(real_rows)}")

    path_counts = Counter(row["path_in_dataset"] for row in raw_echoes_rows)
    duplicate_paths = {path for path, count in path_counts.items() if count > 1}
    excluded_rows = [
        {
            **row,
            "exclusion_reason": "duplicate_path_with_conflicting_original_audio",
        }
        for row in raw_echoes_rows
        if row["path_in_dataset"] in duplicate_paths
    ]
    # A duplicated path can represent only one physical audio file. If its rows
    # point to conflicting references, retaining any one row would be arbitrary.
    for path in duplicate_paths:
        linked_names = {
            row["original_audio"]
            for row in raw_echoes_rows
            if row["path_in_dataset"] == path
        }
        if len(linked_names) == 1:
            raise RuntimeError(
                f"Unexpected exact duplicate rows at {path}; review deduplication policy"
            )
    echoes_rows = [
        row for row in raw_echoes_rows if row["path_in_dataset"] not in duplicate_paths
    ]
    if excluded_rows:
        write_csv(args.excluded_output, excluded_rows)

    validation_by_id = {row["fma_track_id"]: row for row in validation_rows}
    real_by_original = {row["original_audio"]: row for row in real_rows}
    if len(real_by_original) != len(real_rows):
        raise RuntimeError("REAL mapping contains duplicate original_audio values")

    fake_by_original: dict[str, list[dict[str, str]]] = defaultdict(list)
    genres_by_original: dict[str, set[str]] = defaultdict(set)
    for row in echoes_rows:
        fake_by_original[row["original_audio"]].append(row)
        genres_by_original[row["original_audio"]].add(row["genre"])

    fake_names = set(fake_by_original)
    real_names = set(real_by_original)
    if fake_names != real_names:
        raise RuntimeError(
            "REAL/TTA original_audio sets differ: "
            f"fake_only={sorted(fake_names - real_names)}, "
            f"real_only={sorted(real_names - fake_names)}"
        )
    multiple_genres = {
        name: sorted(genres)
        for name, genres in genres_by_original.items()
        if len(genres) != 1
    }
    if multiple_genres:
        raise RuntimeError(f"Reference names with multiple Echoes genres: {multiple_genres}")

    names_by_genre: dict[str, list[str]] = defaultdict(list)
    for original_audio in sorted(real_names):
        genre = next(iter(genres_by_original[original_audio]))
        names_by_genre[genre].append(original_audio)

    rng = random.Random(args.seed)
    split_by_original: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    split_ratios = (0.70, 0.15, 0.15)
    for genre in sorted(names_by_genre):
        names = sorted(names_by_genre[genre])
        rng.shuffle(names)
        counts = allocate_counts(len(names), split_ratios)
        cursor = 0
        for split, count in zip(split_names, counts):
            for original_audio in names[cursor : cursor + count]:
                split_by_original[original_audio] = split
            cursor += count
        if cursor != len(names):
            raise RuntimeError(f"Split allocation failed for genre {genre}")

    group_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    missing_paths: list[str] = []

    for original_audio in sorted(real_names):
        group_id = stable_group_id(original_audio)
        genre = next(iter(genres_by_original[original_audio]))
        split = split_by_original[original_audio]
        real = real_by_original[original_audio]
        track_id = int(real["fma_track_id"])
        validation = validation_by_id.get(str(track_id))
        if not validation or validation["usable_10s"] != "True":
            raise RuntimeError(f"REAL track {track_id} is not validated as usable")

        real_relative_path = real["local_path"]
        real_absolute_path = PROJECT_ROOT / real_relative_path
        if not real_absolute_path.is_file():
            missing_paths.append(str(real_absolute_path))

        fake_rows = fake_by_original[original_audio]
        providers = sorted({row["generator"] for row in fake_rows})
        group_rows.append(
            {
                "group_id": group_id,
                "original_audio": original_audio,
                "genre": genre,
                "split": split,
                "fma_track_id": track_id,
                "real_duration_sec": validation["duration_sec"],
                "tta_track_count": len(fake_rows),
                "tta_generators": "|".join(providers),
            }
        )

        manifest_rows.append(
            {
                "sample_id": f"real_{track_id:06d}",
                "group_id": group_id,
                "original_audio": original_audio,
                "path_root": "project",
                "relative_path": real_relative_path,
                "absolute_path": str(real_absolute_path),
                "label": "REAL",
                "label_id": 0,
                "source": "FMA",
                "generator": "human",
                "generation_type": "REAL",
                "genre": genre,
                "duration_sec": validation["duration_sec"],
                "split": split,
            }
        )

        for fake in sorted(fake_rows, key=lambda row: row["path_in_dataset"]):
            fake_relative_path = fake["path_in_dataset"]
            fake_absolute_path = args.echoes_root / fake_relative_path
            if not fake_absolute_path.is_file():
                missing_paths.append(str(fake_absolute_path))
            manifest_rows.append(
                {
                    "sample_id": stable_fake_id(fake_relative_path),
                    "group_id": group_id,
                    "original_audio": original_audio,
                    "path_root": "echoes",
                    "relative_path": fake_relative_path,
                    "absolute_path": str(fake_absolute_path),
                    "label": "FAKE",
                    "label_id": 1,
                    "source": "Echoes",
                    "generator": fake["generator"],
                    "generation_type": "TTA",
                    "genre": fake["genre"],
                    "duration_sec": fake["duration"],
                    "split": split,
                }
            )

    if missing_paths:
        raise RuntimeError(f"Missing audio paths ({len(missing_paths)}): {missing_paths[:10]}")
    if len({row["sample_id"] for row in manifest_rows}) != len(manifest_rows):
        raise RuntimeError("sample_id values are not unique")

    splits_per_group: dict[str, set[str]] = defaultdict(set)
    labels_per_group: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        splits_per_group[str(row["group_id"])].add(str(row["split"]))
        labels_per_group[str(row["group_id"])].add(str(row["label"]))
    leaking = {group: values for group, values in splits_per_group.items() if len(values) != 1}
    if leaking:
        raise RuntimeError(f"Group leakage detected: {leaking}")
    incomplete = {
        group: values for group, values in labels_per_group.items() if values != {"REAL", "FAKE"}
    }
    if incomplete:
        raise RuntimeError(f"Groups without both labels: {incomplete}")

    manifest_rows.sort(key=lambda row: (split_names.index(str(row["split"])), str(row["group_id"]), int(row["label_id"]), str(row["sample_id"])))
    group_rows.sort(key=lambda row: (split_names.index(str(row["split"])), str(row["genre"]), str(row["group_id"])))
    write_csv(args.output, manifest_rows)
    write_csv(args.groups_output, group_rows)

    summary_rows: list[dict[str, object]] = []
    for split in split_names:
        for label in ("REAL", "FAKE"):
            selected = [
                row for row in manifest_rows if row["split"] == split and row["label"] == label
            ]
            summary_rows.append(
                {
                    "split": split,
                    "label": label,
                    "tracks": len(selected),
                    "unique_groups": len({row["group_id"] for row in selected}),
                    "duration_hours": round(
                        sum(float(row["duration_sec"]) for row in selected) / 3600, 4
                    ),
                }
            )
    write_csv(args.summary_output, summary_rows)

    generator_rows: list[dict[str, object]] = []
    all_generators = sorted({row["generator"] for row in manifest_rows if row["label"] == "FAKE"})
    for split in split_names:
        for generator in all_generators:
            selected = [
                row
                for row in manifest_rows
                if row["split"] == split and row["generator"] == generator
            ]
            generator_rows.append(
                {
                    "split": split,
                    "generator": generator,
                    "tracks": len(selected),
                    "unique_groups": len({row["group_id"] for row in selected}),
                    "duration_hours": round(
                        sum(float(row["duration_sec"]) for row in selected) / 3600, 4
                    ),
                }
            )
    write_csv(args.generator_summary_output, generator_rows)

    group_split_counts = Counter(row["split"] for row in group_rows)
    group_genre_counts = Counter((row["split"], row["genre"]) for row in group_rows)
    track_counts = Counter((row["split"], row["label"]) for row in manifest_rows)
    print(f"Groups:          {len(group_rows)} {dict(group_split_counts)}")
    print(f"Tracks:          {len(manifest_rows)}")
    print(f"TTA raw/used:    {len(raw_echoes_rows)}/{len(echoes_rows)}")
    print(f"TTA excluded:    {len(excluded_rows)} rows at {len(duplicate_paths)} path(s)")
    print(f"Track counts:    {dict(track_counts)}")
    print(f"Group genres:    {dict(group_genre_counts)}")
    print(f"Generators:      {len(all_generators)}")
    print(f"Missing paths:   {len(missing_paths)}")
    print(f"Leaking groups:  {len(leaking)}")
    print(f"Track manifest:  {args.output}")
    print(f"Group split:     {args.groups_output}")
    print(f"Split summary:   {args.summary_output}")
    print(f"Generator stats: {args.generator_summary_output}")
    if excluded_rows:
        print(f"Exclusion log:   {args.excluded_output}")


if __name__ == "__main__":
    main()
