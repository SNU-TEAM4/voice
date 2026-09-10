#!/usr/bin/env python3
"""5단계: 원본/공통 MP3 조건과 metadata-only 기준선을 비교한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-metrics",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ml_baseline" / "metrics.csv",
    )
    parser.add_argument(
        "--controlled-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "ml_codec_control_mp3_64kbps"
        / "metrics.csv",
    )
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "codec_control",
    )
    return parser.parse_args()


def provider_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(metadata), dtype=np.float64)
    real_mask = metadata["label_id"].to_numpy() == 0
    fake_mask = ~real_mask
    weights[real_mask] = 0.5 / int(real_mask.sum())
    generators = sorted(metadata.loc[fake_mask, "generator"].unique())
    for generator in generators:
        mask = fake_mask & metadata["generator"].eq(generator).to_numpy()
        weights[mask] = 0.5 / len(generators) / int(mask.sum())
    weights *= len(weights) / weights.sum()
    return weights


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2), float(thresholds[index])


def evaluate(
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    eer, split_eer_threshold = eer_and_threshold(labels, scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": "Metadata-only Logistic Regression",
        "level": "track",
        "split": split,
        "samples": len(labels),
        "eer": eer,
        "eer_threshold_on_this_split": split_eer_threshold,
        "fixed_validation_threshold": threshold,
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc_fake": average_precision_score(labels, scores),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "real_fpr": fp / max(tn + fp, 1),
        "fake_miss_rate": fn / max(fn + tp, 1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(args.track_manifest)
    validation = pd.read_csv(args.audio_validation)
    technical = validation[
        [
            "sample_id",
            "format_name",
            "codec_name",
            "sample_rate",
            "channels",
            "size_bytes",
            "actual_duration_sec",
        ]
    ]
    metadata = tracks[
        ["sample_id", "group_id", "split", "label", "label_id", "generator", "genre"]
    ].merge(technical, on="sample_id", how="left", validate="one_to_one")
    if metadata.isna().any().any():
        raise RuntimeError("metadata-only 입력에 누락값이 있습니다.")
    metadata["estimated_kbps"] = (
        metadata["size_bytes"] * 8 / metadata["actual_duration_sec"] / 1000
    )

    categorical_columns = ["format_name", "codec_name", "sample_rate", "channels"]
    numeric_columns = ["estimated_kbps"]
    input_columns = categorical_columns + numeric_columns
    train = metadata.loc[metadata["split"].eq("train")].reset_index(drop=True)
    validation_split = metadata.loc[metadata["split"].eq("validation")].reset_index(drop=True)
    test = metadata.loc[metadata["split"].eq("test")].reset_index(drop=True)
    weights = provider_balanced_weights(train)

    selection_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for c_value in [0.1, 1.0, 10.0]:
        model = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                categorical_columns,
                            ),
                            ("numeric", StandardScaler(), numeric_columns),
                        ]
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(
            train[input_columns],
            train["label_id"],
            classifier__sample_weight=weights,
        )
        scores = model.decision_function(validation_split[input_columns])
        eer, threshold = eer_and_threshold(
            validation_split["label_id"].to_numpy(), scores
        )
        row = {
            "C": c_value,
            "validation_eer": eer,
            "validation_threshold": threshold,
            "selected": False,
        }
        selection_rows.append(row)
        if best is None or eer < float(best["eer"]):
            best = {
                "eer": eer,
                "threshold": threshold,
                "C": c_value,
                "model": model,
            }
    if best is None:
        raise RuntimeError("metadata-only 모델 선택에 실패했습니다.")
    for row in selection_rows:
        row["selected"] = float(row["C"]) == float(best["C"])
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "metadata_model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata_model: Pipeline = best["model"]
    metadata_threshold = float(best["threshold"])
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_name, frame in [("validation", validation_split), ("test", test)]:
        scores = metadata_model.decision_function(frame[input_columns])
        metric_rows.append(
            evaluate(
                split_name,
                frame["label_id"].to_numpy(),
                scores,
                metadata_threshold,
            )
        )
        predictions = frame[
            ["sample_id", "group_id", "split", "label", "label_id", "generator", "genre"]
        ].copy()
        predictions["score"] = scores
        predictions["threshold"] = metadata_threshold
        predictions["prediction_id"] = (scores >= metadata_threshold).astype(int)
        predictions["prediction"] = np.where(
            predictions["prediction_id"].eq(1), "FAKE", "REAL"
        )
        prediction_frames.append(predictions)
    metadata_metrics = pd.DataFrame(metric_rows)
    metadata_metrics.to_csv(
        args.output_dir / "metadata_baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "metadata_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(metadata_model, model_dir / "metadata_baseline.joblib")

    raw = pd.read_csv(args.raw_metrics)
    controlled = pd.read_csv(args.controlled_metrics)
    raw_test = raw.loc[raw["split"].eq("test")].copy()
    raw_test.insert(0, "condition", "Original encoding")
    controlled_test = controlled.loc[controlled["split"].eq("test")].copy()
    controlled_test.insert(0, "condition", "Common MP3 64kbps")
    comparison = pd.concat([raw_test, controlled_test], ignore_index=True)
    comparison.to_csv(
        args.output_dir / "audio_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    track_comparison = comparison.loc[comparison["level"].eq("track")].copy()
    pivot = track_comparison.pivot(index="model", columns="condition", values="eer")
    pivot["EER change (percentage points)"] = (
        pivot["Common MP3 64kbps"] - pivot["Original encoding"]
    ) * 100
    pivot.reset_index().to_csv(
        args.output_dir / "track_eer_change.csv", index=False, encoding="utf-8-sig"
    )

    metadata_test = metadata_metrics.loc[metadata_metrics["split"].eq("test")].iloc[0]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    plot_frame = track_comparison.pivot(
        index="model", columns="condition", values="eer"
    ).reindex(["Logistic Regression", "RBF-SVM"])
    (plot_frame * 100).plot.bar(
        ax=axis,
        rot=0,
        color=["#F2CF5B", "#4C78A8"],
    )
    axis.axhline(
        float(metadata_test["eer"]) * 100,
        color="#E45756",
        linestyle="--",
        linewidth=1.5,
        label=f"Metadata-only ({float(metadata_test['eer']) * 100:.2f}%)",
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f%%", padding=3, fontsize=9)
    axis.set(
        title="Test Track EER before and after codec control",
        xlabel="Model",
        ylabel="EER (%) - lower is better",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "track_eer_codec_comparison.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)

    display = track_comparison[
        ["model", "condition", "eer", "roc_auc", "macro_f1", "real_fpr", "fake_miss_rate"]
    ].copy()
    for column in ["eer", "roc_auc", "macro_f1", "real_fpr", "fake_miss_rate"]:
        display[column] = (display[column] * 100).round(2).astype(str) + "%"
    display.columns = [
        "모델",
        "조건",
        "EER",
        "ROC-AUC",
        "Macro-F1",
        "REAL 오탐률",
        "FAKE 놓침률",
    ]

    delta_lr = float(
        pivot.loc["Logistic Regression", "EER change (percentage points)"]
    )
    delta_svm = float(pivot.loc["RBF-SVM", "EER change (percentage points)"])
    report = f"""# 5단계 코덱 통제 실험 결과

## 실험 목적

기존 모델이 AI 생성 특징 대신 REAL과 FAKE의 MP3/WAV, 샘플링레이트,
비트레이트 차이를 이용했는지 확인했다. 기존 데이터와 분할은 변경하지 않았다.

## 비교 조건

- Original encoding: 원본을 mono·24kHz로 디코딩한 기존 조건
- Common MP3 64kbps: 모든 10초 REAL·FAKE 구간을 동일한 MP3 64kbps로
  인코딩한 뒤 다시 디코딩한 통제 조건
- Metadata-only: 음악 내용 없이 원본 형식, 코덱, 샘플링레이트, 채널,
  추정 비트레이트만 사용한 Logistic Regression

## 오디오 모델의 Test 곡 단위 결과

{markdown_table(display)}

- Logistic Regression EER 변화: **{delta_lr:+.2f}%p**
- RBF-SVM EER 변화: **{delta_svm:+.2f}%p**

## Metadata-only 진단

- 선택된 C: {best['C']}
- Validation EER: {float(best['eer']) * 100:.2f}%
- Test EER: {float(metadata_test['eer']) * 100:.2f}%
- Test ROC-AUC: {float(metadata_test['roc_auc']) * 100:.2f}%

음악을 듣지 않고 파일 정보만 사용해도 일정 수준의 구분이 가능하므로 원본
데이터에 인코딩 편향이 존재한다.

## 해석

공통 MP3 처리 후 RBF-SVM의 EER은 상승하여 기존 성능 일부가 코덱 차이에
영향받았음을 보여준다. 반면 Logistic Regression의 EER 변화는 작고, 두
오디오 모델 모두 통제 조건에서 약 9%대 EER을 유지했다. 따라서 기존 성능을
전부 파일 형식 편향만으로 설명할 수는 없으며, 음향 내용에도 분류 가능한
신호가 남아 있다고 해석할 수 있다.

## 한계

공통 MP3 재압축은 최종 형식을 같게 하지만 원본 파일에 이미 남은 과거 압축
흔적을 완전히 제거하지는 못한다. 따라서 이 실험은 편향의 완전 제거가 아니라
코덱 차이를 약화했을 때 성능이 유지되는지를 보는 민감도 분석이다.

## 다음 단계

MusicGen과 Suno를 각각 Train에서 완전히 제외하고, Original과 Common MP3
조건에서 미지 생성기 성능을 비교한다. 이후 같은 분할로 CNN을 학습한다.
"""
    (args.output_dir / "CODEC_CONTROL_결과요약.md").write_text(report, encoding="utf-8")
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "common_codec": "MP3",
                "common_bitrate_kbps": 64,
                "common_sample_rate": 24000,
                "common_channels": 1,
                "metadata_features": input_columns,
                "metadata_selected_C": best["C"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("코덱 통제 비교가 완료되었습니다.")
    print(f"Metadata-only Test EER: {float(metadata_test['eer']):.4f}")
    print(f"Logistic Regression EER 변화: {delta_lr:+.2f}%p")
    print(f"RBF-SVM EER 변화: {delta_svm:+.2f}%p")
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
