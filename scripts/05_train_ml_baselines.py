#!/usr/bin/env python3
"""4단계-B: 수작업 특징으로 Logistic Regression과 RBF-SVM을 학습한다."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "features" / "handcrafted_features.npz",
    )
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ml_baseline",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "segment_id",
        "sample_id",
        "group_id",
        "label",
        "label_id",
        "generator",
        "genre",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"segment manifest에 필요한 열이 없습니다: {sorted(missing)}")
    return frame


def provider_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    """REAL/FAKE 총 가중치를 같게 하고 FAKE 생성기별 총 가중치도 같게 한다."""
    weights = np.zeros(len(metadata), dtype=np.float64)
    real_mask = metadata["label_id"].to_numpy() == 0
    fake_mask = ~real_mask
    real_count = int(real_mask.sum())
    if real_count == 0 or int(fake_mask.sum()) == 0:
        raise ValueError("Train에 REAL과 FAKE가 모두 있어야 합니다.")

    # 전체 가중치의 절반은 REAL에 균등 배분한다.
    weights[real_mask] = 0.5 / real_count

    # 나머지 절반은 생성기마다 먼저 균등 배분한 뒤 각 구간에 나눈다.
    fake_generators = sorted(metadata.loc[fake_mask, "generator"].unique())
    for generator in fake_generators:
        mask = fake_mask & metadata["generator"].eq(generator).to_numpy()
        weights[mask] = 0.5 / len(fake_generators) / int(mask.sum())

    # 평균 가중치를 1로 맞추면 모델의 C 해석이 일반적인 범위에 가까워진다.
    weights *= len(weights) / weights.sum()
    return weights


def continuous_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    return eer, float(thresholds[index])


def aggregate_tracks(metadata: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = metadata[
        ["sample_id", "group_id", "label", "label_id", "generator", "genre", "split"]
    ].copy()
    scored["score"] = scores
    return (
        scored.groupby("sample_id", as_index=False, sort=False)
        .agg(
            group_id=("group_id", "first"),
            label=("label", "first"),
            label_id=("label_id", "first"),
            generator=("generator", "first"),
            genre=("genre", "first"),
            split=("split", "first"),
            score=("score", "mean"),
            segments=("score", "size"),
        )
    )


def metric_row(
    model_name: str,
    level: str,
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
    fixed_threshold: float,
) -> dict[str, object]:
    eer, eer_threshold = eer_and_threshold(labels, scores)
    predictions = (scores >= fixed_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "level": level,
        "split": split,
        "samples": len(labels),
        "eer": eer,
        "eer_threshold_on_this_split": eer_threshold,
        "fixed_validation_threshold": fixed_threshold,
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


def prediction_frame(
    model_name: str,
    level: str,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if level == "segment":
        output = metadata[
            [
                "segment_id",
                "sample_id",
                "group_id",
                "label",
                "label_id",
                "generator",
                "genre",
                "split",
            ]
        ].copy()
        output.insert(0, "item_id", output["segment_id"])
    else:
        output = metadata.copy()
        output.insert(0, "item_id", output["sample_id"])
    output.insert(0, "level", level)
    output.insert(0, "model", model_name)
    output["score"] = scores
    output["threshold"] = threshold
    output["prediction_id"] = (scores >= threshold).astype(int)
    output["prediction"] = np.where(output["prediction_id"].eq(1), "FAKE", "REAL")
    return output


def markdown_table(frame: pd.DataFrame) -> str:
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
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features)
    features = np.asarray(data["X"], dtype=np.float32)
    labels = np.asarray(data["y"], dtype=int)
    segment_ids = data["segment_ids"].astype(str)
    feature_names = data["feature_names"].astype(str)
    metadata = read_manifest(args.segment_manifest)

    if features.shape != (9784, 266):
        raise RuntimeError(f"예상 특징 크기 (9784, 266)과 다릅니다: {features.shape}")
    if metadata["segment_id"].tolist() != segment_ids.tolist():
        raise RuntimeError("특징 행과 segment manifest 순서가 다릅니다.")
    if metadata["label_id"].to_numpy().tolist() != labels.tolist():
        raise RuntimeError("특징 파일과 manifest의 정답 label이 다릅니다.")
    if not np.isfinite(features).all() or len(set(feature_names)) != 266:
        raise RuntimeError("특징에 비정상 값이나 중복 이름이 있습니다.")

    split_indices = {
        split: np.flatnonzero(metadata["split"].eq(split).to_numpy())
        for split in ["train", "validation", "test"]
    }
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    test_indices = split_indices["test"]
    train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
    validation_metadata = metadata.iloc[validation_indices].reset_index(drop=True)
    test_metadata = metadata.iloc[test_indices].reset_index(drop=True)
    train_weights = provider_balanced_weights(train_metadata)

    candidates: list[tuple[str, str, dict[str, object]]] = []
    for c_value in [0.1, 1.0, 10.0]:
        candidates.append(
            ("Logistic Regression", "logistic_regression", {"C": c_value})
        )
    for c_value, gamma in [(1.0, "scale"), (10.0, "scale"), (10.0, 0.001)]:
        candidates.append(("RBF-SVM", "rbf_svm", {"C": c_value, "gamma": gamma}))

    selection_rows: list[dict[str, object]] = []
    best_by_slug: dict[str, dict[str, object]] = {}
    print(
        f"학습 데이터: {len(train_indices):,}개, Validation: {len(validation_indices):,}개",
        flush=True,
    )
    for model_name, slug, parameters in candidates:
        if slug == "logistic_regression":
            estimator = LogisticRegression(
                C=float(parameters["C"]),
                max_iter=3000,
                solver="lbfgs",
                random_state=SEED,
            )
        else:
            estimator = SVC(
                C=float(parameters["C"]),
                gamma=parameters["gamma"],
                kernel="rbf",
                cache_size=2048,
                random_state=SEED,
            )
        model = Pipeline(
            [("scaler", StandardScaler()), ("classifier", estimator)]
        )
        print(f"학습 중: {model_name} {parameters}", flush=True)
        model.fit(
            features[train_indices],
            labels[train_indices],
            classifier__sample_weight=train_weights,
        )
        validation_scores = continuous_scores(model, features[validation_indices])
        segment_eer, segment_threshold = eer_and_threshold(
            labels[validation_indices], validation_scores
        )
        validation_tracks = aggregate_tracks(validation_metadata, validation_scores)
        track_eer, track_threshold = eer_and_threshold(
            validation_tracks["label_id"].to_numpy(),
            validation_tracks["score"].to_numpy(),
        )
        selection_row = {
            "model": model_name,
            "model_slug": slug,
            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            "validation_segment_eer": segment_eer,
            "validation_track_eer": track_eer,
            "validation_segment_threshold": segment_threshold,
            "validation_track_threshold": track_threshold,
            "selected": False,
        }
        selection_rows.append(selection_row)

        current = best_by_slug.get(slug)
        rank = (track_eer, segment_eer)
        if current is None or rank < current["rank"]:
            best_by_slug[slug] = {
                "rank": rank,
                "model_name": model_name,
                "parameters": parameters,
                "model": model,
                "segment_threshold": segment_threshold,
                "track_threshold": track_threshold,
            }
        print(
            f"  Validation EER: segment={segment_eer:.4f}, track={track_eer:.4f}",
            flush=True,
        )

    for row in selection_rows:
        best = best_by_slug[row["model_slug"]]
        row["selected"] = row["parameters"] == json.dumps(
            best["parameters"], ensure_ascii=False, sort_keys=True
        )
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(
        args.output_dir / "model_selection.csv", index=False, encoding="utf-8-sig"
    )

    metric_rows: list[dict[str, object]] = []
    segment_prediction_frames: list[pd.DataFrame] = []
    track_prediction_frames: list[pd.DataFrame] = []
    test_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    confusion_data: dict[tuple[str, str], np.ndarray] = {}
    training_metadata: dict[str, object] = {
        "seed": SEED,
        "feature_file": str(args.features),
        "feature_count": int(features.shape[1]),
        "train_segments": int(len(train_indices)),
        "validation_segments": int(len(validation_indices)),
        "test_segments": int(len(test_indices)),
        "selection_rule": "lowest validation track EER, then segment EER",
        "models": {},
    }

    for slug, best in best_by_slug.items():
        model_name = str(best["model_name"])
        model: Pipeline = best["model"]
        segment_threshold = float(best["segment_threshold"])
        track_threshold = float(best["track_threshold"])
        joblib.dump(model, model_dir / f"{slug}.joblib")
        training_metadata["models"][slug] = {
            "name": model_name,
            "parameters": best["parameters"],
            "validation_segment_threshold": segment_threshold,
            "validation_track_threshold": track_threshold,
        }

        for split, indices, split_metadata in [
            ("validation", validation_indices, validation_metadata),
            ("test", test_indices, test_metadata),
        ]:
            scores = continuous_scores(model, features[indices])
            track_frame = aggregate_tracks(split_metadata, scores)
            segment_labels = labels[indices]
            track_labels = track_frame["label_id"].to_numpy()
            track_scores = track_frame["score"].to_numpy()

            metric_rows.append(
                metric_row(
                    model_name,
                    "segment",
                    split,
                    segment_labels,
                    scores,
                    segment_threshold,
                )
            )
            metric_rows.append(
                metric_row(
                    model_name,
                    "track",
                    split,
                    track_labels,
                    track_scores,
                    track_threshold,
                )
            )
            segment_prediction_frames.append(
                prediction_frame(
                    model_name,
                    "segment",
                    split_metadata,
                    scores,
                    segment_threshold,
                )
            )
            track_prediction_frames.append(
                prediction_frame(
                    model_name,
                    "track",
                    track_frame,
                    track_scores,
                    track_threshold,
                )
            )

            if split == "test":
                for level, level_labels, level_scores, threshold in [
                    ("segment", segment_labels, scores, segment_threshold),
                    ("track", track_labels, track_scores, track_threshold),
                ]:
                    fpr, tpr, _ = roc_curve(level_labels, level_scores)
                    auc = roc_auc_score(level_labels, level_scores)
                    test_curves[(model_name, level)] = (fpr, tpr, auc)
                    predictions = (level_scores >= threshold).astype(int)
                    confusion_data[(model_name, level)] = confusion_matrix(
                        level_labels, predictions, labels=[0, 1]
                    )

    metrics = pd.DataFrame(metric_rows)
    numeric_columns = [
        "eer",
        "eer_threshold_on_this_split",
        "fixed_validation_threshold",
        "roc_auc",
        "pr_auc_fake",
        "macro_f1",
        "balanced_accuracy",
        "real_fpr",
        "fake_miss_rate",
    ]
    metrics[numeric_columns] = metrics[numeric_columns].astype(float)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(segment_prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "segment_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(track_prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "track_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Test ROC 곡선
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"Logistic Regression": "#4C78A8", "RBF-SVM": "#E45756"}
    for axis, level in zip(axes, ["segment", "track"]):
        for model_name in ["Logistic Regression", "RBF-SVM"]:
            fpr, tpr, auc = test_curves[(model_name, level)]
            axis.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=colors[model_name])
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axis.set(
            title=f"Test ROC - {level}",
            xlabel="REAL false positive rate",
            ylabel="FAKE true positive rate",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "test_roc_curves.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    # Validation에서 고정한 threshold로 만든 Test 혼동행렬
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    for row_index, model_name in enumerate(["Logistic Regression", "RBF-SVM"]):
        for column_index, level in enumerate(["segment", "track"]):
            axis = axes[row_index, column_index]
            matrix = confusion_data[(model_name, level)]
            image = axis.imshow(matrix, cmap="Blues")
            for row in range(2):
                for column in range(2):
                    axis.text(
                        column,
                        row,
                        str(matrix[row, column]),
                        ha="center",
                        va="center",
                        color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                        fontsize=12,
                    )
            axis.set(
                title=f"{model_name} - {level}",
                xlabel="Predicted",
                ylabel="Actual",
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["REAL", "FAKE"],
                yticklabels=["REAL", "FAKE"],
            )
    figure.suptitle("Test confusion matrices (Validation threshold)", y=0.995)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "test_confusion_matrices.png", dpi=170, bbox_inches="tight"
    )
    plt.close(figure)

    selected_display = selection_frame.loc[selection_frame["selected"]].copy()
    selected_display["validation_segment_eer"] = (
        selected_display["validation_segment_eer"] * 100
    ).round(2).astype(str) + "%"
    selected_display["validation_track_eer"] = (
        selected_display["validation_track_eer"] * 100
    ).round(2).astype(str) + "%"
    selected_display = selected_display[
        ["model", "parameters", "validation_segment_eer", "validation_track_eer"]
    ]
    selected_display.columns = ["모델", "선택된 설정", "Validation Segment EER", "Validation Track EER"]

    test_display = metrics.loc[metrics["split"].eq("test")].copy()
    for column in [
        "eer",
        "roc_auc",
        "macro_f1",
        "balanced_accuracy",
        "real_fpr",
        "fake_miss_rate",
    ]:
        test_display[column] = (test_display[column] * 100).round(2).astype(str) + "%"
    test_display = test_display[
        [
            "model",
            "level",
            "samples",
            "eer",
            "roc_auc",
            "macro_f1",
            "balanced_accuracy",
            "real_fpr",
            "fake_miss_rate",
        ]
    ]
    test_display.columns = [
        "모델",
        "평가 단위",
        "표본 수",
        "EER",
        "ROC-AUC",
        "Macro-F1",
        "Balanced Accuracy",
        "REAL 오탐률",
        "FAKE 놓침률",
    ]

    test_track = metrics.loc[
        metrics["split"].eq("test") & metrics["level"].eq("track")
    ].sort_values("eer")
    winner = test_track.iloc[0]
    report = f"""# 4단계 기계학습 기준선 결과

## 이 단계에서 한 일

각 10초 구간을 266개의 음향 특징으로 요약한 뒤 Logistic Regression과
RBF-SVM을 학습했다. 266개에는 MFCC, MFCC 변화량, 스펙트럴 중심·대역폭·
rolloff·flatness·contrast, RMS, zero-crossing rate의 평균과 표준편차가 포함된다.

Train만 모델 학습에 사용했고, 모델 설정과 판정 threshold는 Validation으로
선택했다. Test는 선택이 끝난 모델의 최종 성능을 확인할 때만 사용했다.

## 선택된 설정

{markdown_table(selected_display)}

## Test 결과

{markdown_table(test_display)}

- Track EER이 가장 낮은 기계학습 모델: **{winner['model']}**
- 해당 Track EER: **{winner['eer'] * 100:.2f}%**
- 해당 Track ROC-AUC: **{winner['roc_auc'] * 100:.2f}%**

EER은 낮을수록 좋고 ROC-AUC, Macro-F1, Balanced Accuracy는 높을수록 좋다.
Segment는 10초 한 구간의 판정이고 Track은 한 곡의 최대 3개 점수를 평균한
판정이다.

## 클래스 불균형 처리

Train에는 FAKE가 REAL보다 훨씬 많다. 학습 가중치의 절반을 REAL에, 나머지
절반을 FAKE에 배정했고, FAKE 가중치는 다시 12개 생성기에 균등 배분했다.
Validation과 Test의 표본은 삭제하거나 복제하지 않았다.

## 결과를 해석할 때 주의할 점

이 성능은 현재 데이터와 학습에서 본 생성기에 대한 기준선이다. 모델이 실제
AI 생성 흔적뿐 아니라 코덱이나 생성기별 음질을 이용했을 가능성이 있으므로,
아직 실서비스 성능이라고 해석하면 안 된다. 다음 실험에서 MusicGen과 Suno를
각각 학습에서 완전히 제외하여 미지 생성기 일반화를 확인해야 한다.

## 생성 파일

- `model_selection.csv`: Validation 모델 설정 비교
- `metrics.csv`: Validation/Test 상세 지표
- `segment_predictions.csv`: 10초 구간별 예측
- `track_predictions.csv`: 곡별 평균 예측
- `test_roc_curves.png`: Test ROC 곡선
- `test_confusion_matrices.png`: Test 혼동행렬
- `models/*.joblib`: 저장된 최종 모델
"""
    (args.output_dir / "ML_BASELINE_결과요약.md").write_text(report, encoding="utf-8")

    print("기계학습 기준선 학습과 평가가 완료되었습니다.")
    for row in test_track.itertuples(index=False):
        print(
            f"{row.model}: Test Track EER={row.eer:.4f}, "
            f"ROC-AUC={row.roc_auc:.4f}, Macro-F1={row.macro_f1:.4f}"
        )
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
