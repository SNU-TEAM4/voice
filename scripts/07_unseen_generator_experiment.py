#!/usr/bin/env python3
"""6단계: MusicGen·Suno를 학습에서 제외해 미지 생성기 일반화를 평가한다."""

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
HOLDOUT_GENERATORS = ["musicgen", "suno"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--raw-features",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "features" / "handcrafted_features.npz",
    )
    parser.add_argument(
        "--controlled-features",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "features"
        / "handcrafted_features_mp3_64kbps.npz",
    )
    parser.add_argument(
        "--raw-predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ml_baseline",
    )
    parser.add_argument(
        "--controlled-predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "ml_codec_control_mp3_64kbps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "unseen_generator",
    )
    return parser.parse_args()


def provider_balanced_weights(metadata: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(metadata), dtype=np.float64)
    real_mask = metadata["label_id"].to_numpy() == 0
    fake_mask = ~real_mask
    real_count = int(real_mask.sum())
    fake_generators = sorted(metadata.loc[fake_mask, "generator"].unique())
    if real_count == 0 or not fake_generators:
        raise ValueError("학습 데이터에 REAL과 FAKE가 모두 필요합니다.")
    weights[real_mask] = 0.5 / real_count
    for generator in fake_generators:
        mask = fake_mask & metadata["generator"].eq(generator).to_numpy()
        weights[mask] = 0.5 / len(fake_generators) / int(mask.sum())
    weights *= len(weights) / weights.sum()
    return weights


def make_model(slug: str, parameters: dict[str, object]) -> Pipeline:
    if slug == "logistic_regression":
        classifier = LogisticRegression(
            C=float(parameters["C"]),
            max_iter=3000,
            solver="lbfgs",
            random_state=SEED,
        )
    elif slug == "rbf_svm":
        classifier = SVC(
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            kernel="rbf",
            cache_size=2048,
            random_state=SEED,
        )
    else:
        raise ValueError(f"알 수 없는 모델: {slug}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def candidate_parameters(slug: str) -> list[dict[str, object]]:
    if slug == "logistic_regression":
        return [{"C": value} for value in [0.1, 1.0, 10.0]]
    return [
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 10.0, "gamma": 0.001},
    ]


def continuous_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.decision_function(features), dtype=np.float64)


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2), float(thresholds[index])


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


def evaluate(
    condition: str,
    model_name: str,
    holdout: str,
    regime: str,
    level: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    eer, own_threshold = eer_and_threshold(labels, scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "condition": condition,
        "model": model_name,
        "holdout_generator": holdout,
        "regime": regime,
        "level": level,
        "samples": len(labels),
        "real_samples": int(np.sum(labels == 0)),
        "fake_samples": int(np.sum(labels == 1)),
        "eer": eer,
        "eer_threshold_on_test_subset": own_threshold,
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


def standard_seen_rows(
    condition: str,
    predictions_dir: Path,
    model_name: str,
    holdout: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for level, filename in [
        ("segment", "segment_predictions.csv"),
        ("track", "track_predictions.csv"),
    ]:
        predictions = pd.read_csv(predictions_dir / filename)
        subset = predictions.loc[
            predictions["model"].eq(model_name)
            & predictions["split"].eq("test")
            & (predictions["label_id"].eq(0) | predictions["generator"].eq(holdout))
        ].copy()
        thresholds = subset["threshold"].unique()
        if len(thresholds) != 1 or set(subset["label_id"]) != {0, 1}:
            raise RuntimeError(
                f"seen 예측 구성 오류: {condition}, {model_name}, {holdout}, {level}"
            )
        metric_rows.append(
            evaluate(
                condition,
                model_name,
                holdout,
                "seen",
                level,
                subset["label_id"].to_numpy(),
                subset["score"].to_numpy(),
                float(thresholds[0]),
            )
        )
        subset.insert(0, "holdout_generator", holdout)
        subset.insert(0, "regime", "seen")
        subset.insert(0, "condition", condition)
        prediction_frames.append(subset)
    return metric_rows, prediction_frames


def unseen_prediction_frame(
    condition: str,
    model_name: str,
    holdout: str,
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
        output["item_id"] = output["segment_id"]
    else:
        output = metadata.copy()
        output["item_id"] = output["sample_id"]
    output.insert(0, "level", level)
    output.insert(0, "model", model_name)
    output.insert(0, "holdout_generator", holdout)
    output.insert(0, "regime", "unseen")
    output.insert(0, "condition", condition)
    output["score"] = scores
    output["threshold"] = threshold
    output["prediction_id"] = (scores >= threshold).astype(int)
    output["prediction"] = np.where(output["prediction_id"].eq(1), "FAKE", "REAL")
    return output


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

    metadata = pd.read_csv(args.segment_manifest)
    if len(metadata) != 9784 or metadata["segment_id"].duplicated().any():
        raise RuntimeError("segment manifest가 예상한 9,784개 고유 구간이 아닙니다.")

    conditions = [
        (
            "Original encoding",
            "original",
            args.raw_features,
            args.raw_predictions_dir,
        ),
        (
            "Common MP3 64kbps",
            "mp3_64kbps",
            args.controlled_features,
            args.controlled_predictions_dir,
        ),
    ]
    model_specs = [
        ("Logistic Regression", "logistic_regression"),
        ("RBF-SVM", "rbf_svm"),
    ]

    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    run_metadata: dict[str, object] = {
        "seed": SEED,
        "holdout_generators": HOLDOUT_GENERATORS,
        "rule": {
            "train": "REAL + FAKE excluding holdout generator in train split",
            "validation": "REAL + FAKE excluding holdout generator in validation split",
            "test": "REAL + holdout-generator FAKE in test split",
            "selection": "lowest validation track EER, then segment EER",
        },
        "runs": [],
    }

    for condition, condition_slug, feature_path, predictions_dir in conditions:
        data = np.load(feature_path)
        features = np.asarray(data["X"], dtype=np.float32)
        segment_ids = data["segment_ids"].astype(str)
        if features.shape != (9784, 266) or not np.isfinite(features).all():
            raise RuntimeError(f"특징 파일 오류: {feature_path}, shape={features.shape}")
        if metadata["segment_id"].tolist() != segment_ids.tolist():
            raise RuntimeError(f"특징과 manifest 순서 불일치: {feature_path}")

        for holdout in HOLDOUT_GENERATORS:
            train_mask = metadata["split"].eq("train") & ~metadata["generator"].eq(holdout)
            validation_mask = metadata["split"].eq("validation") & ~metadata["generator"].eq(holdout)
            test_mask = metadata["split"].eq("test") & (
                metadata["label_id"].eq(0) | metadata["generator"].eq(holdout)
            )
            train_indices = np.flatnonzero(train_mask.to_numpy())
            validation_indices = np.flatnonzero(validation_mask.to_numpy())
            test_indices = np.flatnonzero(test_mask.to_numpy())
            train_metadata = metadata.iloc[train_indices].reset_index(drop=True)
            validation_metadata = metadata.iloc[validation_indices].reset_index(drop=True)
            test_metadata = metadata.iloc[test_indices].reset_index(drop=True)

            if holdout in set(train_metadata["generator"]) or holdout in set(
                validation_metadata["generator"]
            ):
                raise RuntimeError(f"{holdout}이 Train 또는 Validation에 남아 있습니다.")
            if set(test_metadata.loc[test_metadata["label_id"].eq(1), "generator"]) != {
                holdout
            }:
                raise RuntimeError(f"Test FAKE가 {holdout}만 포함하지 않습니다.")
            weights = provider_balanced_weights(train_metadata)

            print(
                f"[{condition}] holdout={holdout}: train={len(train_indices):,}, "
                f"validation={len(validation_indices):,}, test={len(test_indices):,}",
                flush=True,
            )
            for model_name, model_slug in model_specs:
                best: dict[str, object] | None = None
                for parameters in candidate_parameters(model_slug):
                    model = make_model(model_slug, parameters)
                    model.fit(
                        features[train_indices],
                        train_metadata["label_id"].to_numpy(),
                        classifier__sample_weight=weights,
                    )
                    validation_scores = continuous_scores(
                        model, features[validation_indices]
                    )
                    segment_eer, segment_threshold = eer_and_threshold(
                        validation_metadata["label_id"].to_numpy(),
                        validation_scores,
                    )
                    validation_tracks = aggregate_tracks(
                        validation_metadata, validation_scores
                    )
                    track_eer, track_threshold = eer_and_threshold(
                        validation_tracks["label_id"].to_numpy(),
                        validation_tracks["score"].to_numpy(),
                    )
                    row = {
                        "condition": condition,
                        "holdout_generator": holdout,
                        "model": model_name,
                        "model_slug": model_slug,
                        "parameters": json.dumps(
                            parameters, ensure_ascii=False, sort_keys=True
                        ),
                        "train_segments": len(train_indices),
                        "validation_segments": len(validation_indices),
                        "validation_segment_eer": segment_eer,
                        "validation_track_eer": track_eer,
                        "validation_segment_threshold": segment_threshold,
                        "validation_track_threshold": track_threshold,
                        "selected": False,
                    }
                    selection_rows.append(row)
                    rank = (track_eer, segment_eer)
                    if best is None or rank < best["rank"]:
                        best = {
                            "rank": rank,
                            "model": model,
                            "parameters": parameters,
                            "segment_threshold": segment_threshold,
                            "track_threshold": track_threshold,
                        }
                if best is None:
                    raise RuntimeError(f"모델 선택 실패: {condition}, {holdout}, {model_name}")

                selected_parameters = json.dumps(
                    best["parameters"], ensure_ascii=False, sort_keys=True
                )
                for row in selection_rows:
                    if (
                        row["condition"] == condition
                        and row["holdout_generator"] == holdout
                        and row["model"] == model_name
                        and row["parameters"] == selected_parameters
                    ):
                        row["selected"] = True

                model: Pipeline = best["model"]
                path = model_dir / f"{condition_slug}__{model_slug}__holdout_{holdout}.joblib"
                joblib.dump(model, path)
                test_scores = continuous_scores(model, features[test_indices])
                test_tracks = aggregate_tracks(test_metadata, test_scores)
                segment_threshold = float(best["segment_threshold"])
                track_threshold = float(best["track_threshold"])
                metric_rows.append(
                    evaluate(
                        condition,
                        model_name,
                        holdout,
                        "unseen",
                        "segment",
                        test_metadata["label_id"].to_numpy(),
                        test_scores,
                        segment_threshold,
                    )
                )
                metric_rows.append(
                    evaluate(
                        condition,
                        model_name,
                        holdout,
                        "unseen",
                        "track",
                        test_tracks["label_id"].to_numpy(),
                        test_tracks["score"].to_numpy(),
                        track_threshold,
                    )
                )
                prediction_frames.append(
                    unseen_prediction_frame(
                        condition,
                        model_name,
                        holdout,
                        "segment",
                        test_metadata,
                        test_scores,
                        segment_threshold,
                    )
                )
                prediction_frames.append(
                    unseen_prediction_frame(
                        condition,
                        model_name,
                        holdout,
                        "track",
                        test_tracks,
                        test_tracks["score"].to_numpy(),
                        track_threshold,
                    )
                )
                run_metadata["runs"].append(
                    {
                        "condition": condition,
                        "holdout_generator": holdout,
                        "model": model_name,
                        "selected_parameters": best["parameters"],
                        "train_segments": len(train_indices),
                        "validation_segments": len(validation_indices),
                        "test_segments": len(test_indices),
                        "model_path": str(path),
                    }
                )

                seen_metrics, seen_predictions = standard_seen_rows(
                    condition, predictions_dir, model_name, holdout
                )
                metric_rows.extend(seen_metrics)
                prediction_frames.extend(seen_predictions)

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(
        args.output_dir / "validation_model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "holdout_generator", "model", "level", "regime"]
    )
    metrics.to_csv(
        args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(prediction_frames, ignore_index=True, sort=False).to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    track_metrics = metrics.loc[metrics["level"].eq("track")].copy()
    gap = track_metrics.pivot_table(
        index=["condition", "model", "holdout_generator"],
        columns="regime",
        values=["eer", "roc_auc", "real_fpr", "fake_miss_rate"],
    )
    gap.columns = [f"{metric}_{regime}" for metric, regime in gap.columns]
    gap = gap.reset_index()
    gap["eer_gap"] = gap["eer_unseen"] - gap["eer_seen"]
    gap["roc_auc_change"] = gap["roc_auc_unseen"] - gap["roc_auc_seen"]
    gap.to_csv(
        args.output_dir / "track_generalization_gap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2개 생성기 × 2개 오디오 조건을 한눈에 보는 곡 EER 그래프
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    condition_order = ["Original encoding", "Common MP3 64kbps"]
    model_order = ["Logistic Regression", "RBF-SVM"]
    for row_index, holdout in enumerate(HOLDOUT_GENERATORS):
        for column_index, condition in enumerate(condition_order):
            axis = axes[row_index, column_index]
            subset = track_metrics.loc[
                track_metrics["holdout_generator"].eq(holdout)
                & track_metrics["condition"].eq(condition)
            ]
            pivot = subset.pivot(index="model", columns="regime", values="eer").reindex(
                model_order
            )
            (pivot[["seen", "unseen"]] * 100).plot.bar(
                ax=axis,
                color=["#4C78A8", "#E45756"],
                rot=0,
                legend=False,
            )
            for container in axis.containers:
                axis.bar_label(container, fmt="%.2f%%", padding=2, fontsize=8)
            axis.set(
                title=f"Holdout {holdout} - {condition}",
                xlabel="",
                ylabel="Track EER (%)" if column_index == 0 else "",
            )
            axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Seen vs unseen generator performance", fontsize=15)
    figure.tight_layout(rect=[0, 0.05, 1, 0.96])
    figure.savefig(
        args.output_dir / "seen_unseen_track_eer.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)

    display = gap[
        [
            "condition",
            "model",
            "holdout_generator",
            "eer_seen",
            "eer_unseen",
            "eer_gap",
            "roc_auc_seen",
            "roc_auc_unseen",
        ]
    ].copy()
    for column in [
        "eer_seen",
        "eer_unseen",
        "eer_gap",
        "roc_auc_seen",
        "roc_auc_unseen",
    ]:
        display[column] = (display[column] * 100).round(2).astype(str) + "%"
    display.columns = [
        "조건",
        "모델",
        "제외 생성기",
        "Seen EER",
        "Unseen EER",
        "EER 격차",
        "Seen ROC-AUC",
        "Unseen ROC-AUC",
    ]
    worst = gap.sort_values("eer_gap", ascending=False).iloc[0]
    best = gap.sort_values("eer_gap", ascending=True).iloc[0]
    report = f"""# 6단계 미지 생성기 일반화 실험 결과

## 실험 질문

MusicGen 또는 Suno 음악을 학습에서 한 번도 보여주지 않아도 AI 생성 음악으로
탐지할 수 있는지 확인했다.

## 데이터 구성

- Train: 기존 train의 REAL + holdout 생성기를 제외한 FAKE
- Validation: 기존 validation의 REAL + holdout 생성기를 제외한 FAKE
- Test: 기존 test의 REAL + holdout 생성기 FAKE만 사용
- 동일 원곡 그룹 분할과 동일 10초 구간 유지
- Original encoding과 Common MP3 64kbps 조건 모두 평가

`seen`은 해당 생성기를 학습에 포함한 기존 모델이고, `unseen`은 해당 생성기를
Train과 Validation에서 완전히 제외한 새 모델이다. 양쪽을 같은 Test 하위집합에
평가했다.

## Test 곡 단위 결과

{markdown_table(display)}

EER 격차는 `Unseen EER - Seen EER`이다. 양수이면 처음 보는 생성기에서
성능이 나빠졌다는 뜻이고, 0에 가까우면 학습에 없던 생성기에도 비교적 잘
일반화했다는 뜻이다.

## 관찰

- 가장 큰 성능 저하: **{worst['condition']} / {worst['model']} / {worst['holdout_generator']}**, {worst['eer_gap'] * 100:+.2f}%p
- 가장 작은 변화: **{best['condition']} / {best['model']} / {best['holdout_generator']}**, {best['eer_gap'] * 100:+.2f}%p

생성기를 제외했는데 EER이 낮아지는 경우도 있을 수 있다. 이는 탐지기가 더
좋아졌다는 확정 증거가 아니라, Test REAL 44곡과 생성기별 FAKE 40여 곡으로
표본이 작고 학습 구성과 threshold가 함께 달라졌기 때문에 생기는 변동으로
해석해야 한다.

## 한계와 다음 단계

두 생성기만으로 모든 미래 생성기를 대표할 수 없고, 원본 코덱 흔적도 완전히
제거되지는 않는다. 다음으로 같은 데이터 분할에 Log-Mel CNN을 학습해 수작업
특징 모델과 딥러닝 모델의 표준·미지 생성기 성능을 비교한다.
"""
    (args.output_dir / "UNSEEN_GENERATOR_결과요약.md").write_text(
        report, encoding="utf-8"
    )

    print("미지 생성기 일반화 실험이 완료되었습니다.")
    print(gap[["condition", "model", "holdout_generator", "eer_seen", "eer_unseen", "eer_gap"]].to_string(index=False))
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
