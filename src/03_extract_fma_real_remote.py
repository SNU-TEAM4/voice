#!/usr/bin/env python3
"""
Extract only the selected FMA REAL 30-second clips from the remote fma_large.zip.

The remote ZIP server supports HTTP Range requests, so this does NOT download
the full ~93 GB archive. Only the requested MP3 entries are transferred.

Default input:
  data/metadata/fma_real_mapping.csv

Default output:
  data/raw/FMA/selected_30s/<folder>/<track_id>.mp3
  data/metadata/fma_remote_extract_report.csv

Examples:
  # Small test
  python src/03_extract_fma_real_remote.py --limit 5

  # Full 296 tracks
  python src/03_extract_fma_real_remote.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip


REMOTE_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"


def find_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for p in candidates:
        if (p / "data").exists():
            return p.resolve()
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. project 폴더에서 실행하세요."
    )


def member_name(track_id: int) -> str:
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return f"fma_large/{folder}/{filename}"


def output_path(base_dir: Path, track_id: int) -> Path:
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return base_dir / folder / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서 N곡만 추출합니다. 예: --limit 5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 파일도 다시 다운로드합니다.",
    )
    args = parser.parse_args()

    project_root = find_project_root()

    mapping_path = project_root / "data/metadata/fma_real_mapping.csv"
    output_dir = project_root / "data/raw/FMA/selected_30s"
    report_path = project_root / "data/metadata/fma_remote_extract_report.csv"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"{mapping_path} 가 없습니다.\n"
            "먼저 fma_real_mapping.csv를 생성하세요."
        )

    mapping = pd.read_csv(mapping_path)

    required = {"track_id", "original_audio", "genre"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"mapping CSV에 필요한 컬럼이 없습니다: {sorted(missing)}"
        )

    work = mapping.copy()
    work["track_id"] = work["track_id"].astype(int)

    if args.limit is not None:
        work = work.head(args.limit).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Remote ZIP :", REMOTE_ZIP_URL)
    print("Tracks     :", len(work))
    print("Output     :", output_dir)
    print("Report     :", report_path)
    print()

    results = []

    with RemoteZip(REMOTE_ZIP_URL) as z:
        print("Remote ZIP 연결 성공\n")

        for i, row in enumerate(work.itertuples(index=False), start=1):
            tid = int(row.track_id)
            member = member_name(tid)
            dst = output_path(output_dir, tid)

            print(f"[{i:03d}/{len(work):03d}] track_id={tid} | {row.original_audio}")

            status = ""
            ok = False
            file_size = None
            compressed_size = None
            elapsed_sec = None

            try:
                if dst.exists() and dst.stat().st_size > 0 and not args.overwrite:
                    info = z.getinfo(member)
                    file_size = info.file_size
                    compressed_size = info.compress_size

                    # Existing local file should match the ZIP entry size.
                    if dst.stat().st_size == file_size:
                        status = "already_exists"
                        ok = True
                        print(f"  status : {status}")
                        print(f"  size   : {dst.stat().st_size:,} bytes")
                    else:
                        print(
                            "  기존 파일 크기가 ZIP entry와 달라 다시 다운로드합니다."
                        )
                        dst.unlink()

                if not ok:
                    start = time.time()

                    info = z.getinfo(member)
                    file_size = info.file_size
                    compressed_size = info.compress_size

                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_suffix(".mp3.part")

                    if tmp.exists():
                        tmp.unlink()

                    with z.open(member) as src, open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)

                    local_size = tmp.stat().st_size

                    if local_size != file_size:
                        tmp.unlink(missing_ok=True)
                        raise IOError(
                            f"파일 크기 불일치: expected={file_size}, got={local_size}"
                        )

                    tmp.replace(dst)

                    elapsed_sec = round(time.time() - start, 2)
                    status = "downloaded"
                    ok = True

                    print(f"  status : {status}")
                    print(f"  size   : {dst.stat().st_size:,} bytes")
                    print(f"  time   : {elapsed_sec:.2f} sec")

            except KeyError:
                status = "not_found_in_zip"
                print("  status : NOT FOUND IN ZIP")

            except Exception as e:
                status = f"{type(e).__name__}: {e}"
                print(f"  status : FAILED - {status}")

            results.append({
                "track_id": tid,
                "original_audio": row.original_audio,
                "genre": row.genre,
                "zip_member": member,
                "local_path": str(dst.relative_to(project_root)),
                "file_size": file_size,
                "compressed_size": compressed_size,
                "elapsed_sec": elapsed_sec,
                "ok": ok,
                "status": status,
            })

            # Save after every track so interrupted runs are still auditable.
            pd.DataFrame(results).to_csv(
                report_path,
                index=False,
                encoding="utf-8-sig",
            )

    report = pd.DataFrame(results)

    success = int(report["ok"].sum())
    failed = int((~report["ok"]).sum())

    print("\n===== SUMMARY =====")
    print("Requested :", len(report))
    print("Success   :", success)
    print("Failed    :", failed)

    if success:
        total_bytes = int(
            report.loc[report["ok"], "file_size"]
            .fillna(0)
            .sum()
        )
        print(
            "Total size:",
            f"{total_bytes / (1024 ** 2):.2f} MB"
        )

    print("Report    :", report_path)

    failures = report[~report["ok"]]

    if len(failures):
        print("\n===== FAILURES =====")
        print(
            failures[
                ["track_id", "original_audio", "status", "zip_member"]
            ].to_string(index=False)
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
