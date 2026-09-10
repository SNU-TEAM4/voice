#!/usr/bin/env python3
"""Build a reproducible Echoes-to-FMA real-track mapping using only stdlib."""

from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECHOES_ROOT = Path(os.environ.get("ECHOES_ROOT", PROJECT_ROOT / "data" / "Echoes"))
FMA_METADATA_ROOT = Path(
    os.environ.get("FMA_METADATA_ROOT", PROJECT_ROOT / "data" / "fma_metadata")
)
# These two FMA excerpts are 1.6 KB placeholder/corrupt MP3s in fma_large.
# Their same-title/artist alternatives decode normally and are selected instead.
KNOWN_BAD_TRACK_IDS = {148786, 148788}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return re.sub(r"\s+", " ", text).strip()


def preferred_license(license_name: str) -> bool:
    """Match the CC0, CC-BY, or public-domain policy stated by Echoes."""
    value = license_name.casefold().strip()
    if "public domain" in value or "cc0" in value:
        return True
    excluded = (
        "noncommercial",
        "non-commercial",
        "noderiv",
        "no-deriv",
        "no derivative",
        "sharealike",
        "share alike",
    )
    attribution = (
        value == "attribution"
        or value.startswith("attribution ")
        or "creative commons attribution" in value
    )
    return attribution and not any(term in value for term in excluded)


def read_echoes(path: Path) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["original_audio"]
            item = grouped.setdefault(
                name,
                {"genres": set(), "tta_count": 0, "ata_count": 0},
            )
            item["genres"].add(row["genre"])
            if row["type"] == "TTA":
                item["tta_count"] += 1
            elif row["type"] == "ATA":
                item["ata_count"] += 1
    return grouped


def read_fma(path: Path) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        upper = next(reader)
        lower = next(reader)
        next(reader)  # row containing only the track_id index label
        index = {(a, b): i for i, (a, b) in enumerate(zip(upper, lower))}
        required = [
            ("artist", "name"),
            ("set", "subset"),
            ("track", "bit_rate"),
            ("track", "duration"),
            ("track", "genre_top"),
            ("track", "license"),
            ("track", "title"),
        ]
        missing = [key for key in required if key not in index]
        if missing:
            raise ValueError(f"FMA tracks.csv is missing columns: {missing}")

        for row in reader:
            if not row:
                continue
            track_id = int(row[0])
            title = row[index[("track", "title")]]
            artist = row[index[("artist", "name")]]
            duration_text = row[index[("track", "duration")]]
            bit_rate_text = row[index[("track", "bit_rate")]]
            record = {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "subset": row[index[("set", "subset")]],
                "genre_top": row[index[("track", "genre_top")]],
                "license": row[index[("track", "license")]],
                "duration": float(duration_text) if duration_text else 0.0,
                "bit_rate": int(float(bit_rate_text)) if bit_rate_text else 0,
            }
            grouped[normalize(f"{title} - {artist}")].append(record)
    return grouped


def candidate_rank(
    candidate: dict[str, object], echo_genres: set[str]
) -> tuple[int, int, int, int]:
    known_bad = int(int(candidate["track_id"]) in KNOWN_BAD_TRACK_IDS)
    genre_match = int(str(candidate["genre_top"]) in echo_genres)
    license_match = int(preferred_license(str(candidate["license"])))
    # Decodability wins; then prefer genre, stated license policy, and lowest ID.
    return (known_bad, -genre_match, -license_match, int(candidate["track_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--echoes-manifest",
        type=Path,
        default=ECHOES_ROOT / "dataset_manifest.csv",
    )
    parser.add_argument(
        "--fma-tracks",
        type=Path,
        default=FMA_METADATA_ROOT / "tracks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "real_track_map.csv",
    )
    parser.add_argument(
        "--ambiguities",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ambiguous_matches.csv",
    )
    args = parser.parse_args()

    echoes = read_echoes(args.echoes_manifest)
    fma = read_fma(args.fma_tracks)

    selected_rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    unmatched: list[str] = []

    for original_audio in sorted(echoes, key=normalize):
        echo_info = echoes[original_audio]
        echo_genres = set(echo_info["genres"])
        candidates = fma.get(normalize(original_audio), [])
        if not candidates:
            unmatched.append(original_audio)
            continue

        candidates = sorted(candidates, key=lambda item: candidate_rank(item, echo_genres))
        selected = candidates[0]
        track_id = int(selected["track_id"])
        estimated_seconds = min(float(selected["duration"]), 30.0)
        estimated_bytes = round(int(selected["bit_rate"]) * estimated_seconds / 8)
        genre_match = str(selected["genre_top"]) in echo_genres
        license_match = preferred_license(str(selected["license"]))

        excluded_bad_candidate = any(
            int(candidate["track_id"]) in KNOWN_BAD_TRACK_IDS for candidate in candidates
        )
        if excluded_bad_candidate:
            rule = "exclude_corrupt_then_genre_license_track_id"
        elif len(candidates) == 1:
            rule = "unique_exact_title_artist"
        else:
            rule = "genre_then_license_then_lowest_track_id"

        selected_rows.append(
            {
                "original_audio": original_audio,
                "echo_genre": "|".join(sorted(echo_genres)),
                "tta_count": echo_info["tta_count"],
                "ata_count": echo_info["ata_count"],
                "fma_track_id": track_id,
                "fma_title": selected["title"],
                "fma_artist": selected["artist"],
                "fma_genre_top": selected["genre_top"],
                "fma_subset": selected["subset"],
                "fma_license": selected["license"],
                "fma_duration_sec": selected["duration"],
                "fma_bit_rate": selected["bit_rate"],
                "genre_match": genre_match,
                "preferred_license": license_match,
                "candidate_count": len(candidates),
                "selection_rule": rule,
                "archive_path": f"fma_large/{track_id // 1000:03d}/{track_id:06d}.mp3",
                "local_path": f"data/fma_real/{track_id:06d}.mp3",
                "estimated_30s_bytes": estimated_bytes,
            }
        )

        if len(candidates) > 1:
            for rank, candidate in enumerate(candidates, start=1):
                ambiguity_rows.append(
                    {
                        "original_audio": original_audio,
                        "echo_genre": "|".join(sorted(echo_genres)),
                        "candidate_rank": rank,
                        "selected": rank == 1,
                        "known_bad": int(candidate["track_id"]) in KNOWN_BAD_TRACK_IDS,
                        "fma_track_id": candidate["track_id"],
                        "fma_title": candidate["title"],
                        "fma_artist": candidate["artist"],
                        "fma_genre_top": candidate["genre_top"],
                        "fma_subset": candidate["subset"],
                        "fma_license": candidate["license"],
                        "fma_duration_sec": candidate["duration"],
                        "fma_bit_rate": candidate["bit_rate"],
                    }
                )

    if unmatched:
        raise RuntimeError(f"Unmatched Echoes references ({len(unmatched)}): {unmatched}")
    if len(selected_rows) != len(echoes):
        raise RuntimeError("Mapping row count does not match Echoes reference count")
    if len({row["fma_track_id"] for row in selected_rows}) != len(selected_rows):
        raise RuntimeError("The selected FMA track IDs are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    args.ambiguities.parent.mkdir(parents=True, exist_ok=True)
    with args.ambiguities.open("w", encoding="utf-8", newline="") as handle:
        if ambiguity_rows:
            writer = csv.DictWriter(handle, fieldnames=list(ambiguity_rows[0]))
            writer.writeheader()
            writer.writerows(ambiguity_rows)

    subset_counts = Counter(str(row["fma_subset"]) for row in selected_rows)
    total_bytes = sum(int(row["estimated_30s_bytes"]) for row in selected_rows)
    print(f"Echoes reference names: {len(echoes)}")
    print(f"Selected FMA tracks:    {len(selected_rows)}")
    print(f"Ambiguous names:        {sum(int(row['candidate_count']) > 1 for row in selected_rows)}")
    print(f"Genre mismatches:       {sum(row['genre_match'] is False for row in selected_rows)}")
    print(f"Non-preferred licenses: {sum(row['preferred_license'] is False for row in selected_rows)}")
    print(f"FMA subset counts:      {dict(subset_counts)}")
    print(f"Estimated MP3 size:     {total_bytes / 1024 / 1024:.1f} MiB")
    print(f"Mapping:                {args.output}")
    print(f"Ambiguity report:       {args.ambiguities}")


if __name__ == "__main__":
    main()
