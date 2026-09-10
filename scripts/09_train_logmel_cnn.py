#!/usr/bin/env python3
"""7단계-B: 공통 MP3 Log-Mel 입력으로 경량 CNN을 학습하고 평가한다."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logmel-cache",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "features"
        / "logmel_128x249_mp3_64kbps.npy",
    )
    parser.add_argument(
        "--segment-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "segment_manifest.csv",
    )
    parser.add_argument(
        "--ml-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "ml_codec_control_mp3_64kbps"
        / "metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cnn_mp3_64kbps",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="실제 캐시를 읽지 않고 무작위 입력으로 순전파·역전파만 검사합니다.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS를 요청했지만 현재 실행 환경에서 사용할 수 없습니다.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LogMelDataset(Dataset):
    def __init__(
        self,
        cache: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        augment: bool,
    ) -> None:
        self.cache = cache
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        absolute_index = int(self.indices[item])
        values = np.asarray(self.cache[absolute_index], dtype=np.float32).copy()
        # 입력 dB 범위 [-80, 0]을 약 [-1, 1]로 바꾼다.
        values = (values + 40.0) / 40.0
        if self.augment:
            if np.random.random() < 0.5:
                width = int(np.random.randint(1, 13))
                start = int(np.random.randint(0, values.shape[0] - width + 1))
                values[start : start + width, :] = -1.0
            if np.random.random() < 0.5:
                width = int(np.random.randint(1, 26))
                start = int(np.random.randint(0, values.shape[1] - width + 1))
                values[:, start : start + width] = -1.0
        tensor = torch.from_numpy(values).unsqueeze(0)
        label = torch.tensor(float(self.labels[absolute_index]), dtype=torch.float32)
        return tensor, label, absolute_index


class SmallLogMelCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [1, 16, 32, 64, 128]
        blocks = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.30), nn.Linear(128, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.features(inputs)
        pooled = self.pool(hidden)
        return self.classifier(pooled).squeeze(1)


def provider_balanced_sampler_weights(metadata: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(metadata), dtype=np.float64)
    real_mask = metadata["label_id"].to_numpy() == 0
    fake_mask = ~real_mask
    weights[real_mask] = 0.5 / int(real_mask.sum())
    generators = sorted(metadata.loc[fake_mask, "generator"].unique())
    for generator in generators:
        mask = fake_mask & metadata["generator"].eq(generator).to_numpy()
        weights[mask] = 0.5 / len(generators) / int(mask.sum())
    return weights


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2), float(thresholds[index])


def aggregate_tracks(metadata: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    frame = metadata[
        ["sample_id", "group_id", "label", "label_id", "generator", "genre", "split"]
    ].copy()
    frame["score"] = scores
    return (
        frame.groupby("sample_id", as_index=False, sort=False)
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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    labels_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels, indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = loss_function(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
            labels_parts.append(labels.detach().cpu().numpy())
            scores_parts.append(logits.detach().cpu().numpy())
            indices_parts.append(indices.numpy())
    return (
        total_loss / max(total_items, 1),
        np.concatenate(labels_parts).astype(int),
        np.concatenate(scores_parts).astype(np.float64),
        np.concatenate(indices_parts).astype(int),
    )


def metric_row(
    level: str,
    split: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    eer, own_threshold = eer_and_threshold(labels, scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": "Log-Mel CNN",
        "condition": "Common MP3 64kbps",
        "level": level,
        "split": split,
        "samples": len(labels),
        "eer": eer,
        "eer_threshold_on_this_split": own_threshold,
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


def prediction_frame(
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
    output.insert(0, "model", "Log-Mel CNN")
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
    set_seed(SEED)
    device = choose_device(args.device)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = SmallLogMelCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"PyTorch={torch.__version__}, device={device}, parameters={parameter_count:,}",
        flush=True,
    )

    if args.smoke_test:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        inputs = torch.randn(8, 1, 128, 249, device=device)
        labels = torch.randint(0, 2, (8,), device=device).float()
        logits = model(inputs)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if logits.shape != (8,) or not torch.isfinite(loss).item():
            raise RuntimeError("CNN smoke test 실패")
        print(f"CNN smoke test=PASS, output={tuple(logits.shape)}, loss={loss.item():.4f}")
        return

    if not args.logmel_cache.is_file():
        raise FileNotFoundError(f"Log-Mel 캐시가 없습니다: {args.logmel_cache}")
    cache_metadata_path = args.logmel_cache.with_suffix(".json")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(args.segment_manifest)
    segment_hash = hashlib.sha256(
        "\n".join(metadata["segment_id"].astype(str)).encode()
    ).hexdigest()
    if cache_metadata["segment_id_sha256"] != segment_hash:
        raise RuntimeError("Log-Mel 캐시와 segment manifest의 순서가 다릅니다.")
    cache = np.load(args.logmel_cache, mmap_mode="r")
    if cache.shape != (9784, 128, 249) or cache.dtype != np.float16:
        raise RuntimeError(f"예상하지 못한 Log-Mel 캐시: {cache.shape}, {cache.dtype}")
    labels = metadata["label_id"].to_numpy(dtype=int)

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

    local_weights = provider_balanced_sampler_weights(train_metadata)
    sampler_generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(local_weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        LogMelDataset(cache, labels, train_indices, augment=True),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )
    validation_loader = DataLoader(
        LogMelDataset(cache, labels, validation_indices, augment=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        LogMelDataset(cache, labels, test_indices, augment=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate / 20
    )
    history: list[dict[str, object]] = []
    best_rank: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for inputs, batch_labels, _ in train_loader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_labels)
            train_items += len(batch_labels)
        scheduler.step()

        validation_loss, validation_labels, validation_scores, validation_order = (
            evaluate_loader(model, validation_loader, device, loss_function)
        )
        if not np.array_equal(validation_order, validation_indices):
            raise RuntimeError("Validation 순서가 manifest와 다릅니다.")
        segment_eer, _ = eer_and_threshold(validation_labels, validation_scores)
        validation_tracks = aggregate_tracks(validation_metadata, validation_scores)
        track_eer, _ = eer_and_threshold(
            validation_tracks["label_id"].to_numpy(),
            validation_tracks["score"].to_numpy(),
        )
        validation_auc = roc_auc_score(validation_labels, validation_scores)
        train_loss = train_loss_sum / max(train_items, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_segment_eer": segment_eer,
                "validation_track_eer": track_eer,
                "validation_segment_roc_auc": validation_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        rank = (track_eer, segment_eer)
        improved = best_rank is None or rank < best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:02d}: train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}, segment_EER={segment_eer:.4f}, "
            f"track_EER={track_eer:.4f}, best={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: {args.patience} epoch 동안 개선 없음", flush=True)
            break

    if best_state is None or best_rank is None:
        raise RuntimeError("최적 CNN state를 선택하지 못했습니다.")
    model.load_state_dict(best_state)
    model.to(device)

    # 최적 epoch 모델로 Validation threshold를 고정하고 Test를 한 번 평가한다.
    validation_loss, validation_labels, validation_scores, validation_order = evaluate_loader(
        model, validation_loader, device, loss_function
    )
    validation_tracks = aggregate_tracks(validation_metadata, validation_scores)
    _, segment_threshold = eer_and_threshold(validation_labels, validation_scores)
    _, track_threshold = eer_and_threshold(
        validation_tracks["label_id"].to_numpy(),
        validation_tracks["score"].to_numpy(),
    )
    test_loss, test_labels, test_scores, test_order = evaluate_loader(
        model, test_loader, device, loss_function
    )
    if not np.array_equal(test_order, test_indices):
        raise RuntimeError("Test 순서가 manifest와 다릅니다.")
    test_tracks = aggregate_tracks(test_metadata, test_scores)

    metric_rows = [
        metric_row(
            "segment",
            "validation",
            validation_labels,
            validation_scores,
            segment_threshold,
        ),
        metric_row(
            "track",
            "validation",
            validation_tracks["label_id"].to_numpy(),
            validation_tracks["score"].to_numpy(),
            track_threshold,
        ),
        metric_row(
            "segment", "test", test_labels, test_scores, segment_threshold
        ),
        metric_row(
            "track",
            "test",
            test_tracks["label_id"].to_numpy(),
            test_tracks["score"].to_numpy(),
            track_threshold,
        ),
    ]
    metrics = pd.DataFrame(metric_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        args.output_dir / "history.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [
            prediction_frame(
                "segment", validation_metadata, validation_scores, segment_threshold
            ),
            prediction_frame("segment", test_metadata, test_scores, segment_threshold),
        ],
        ignore_index=True,
    ).to_csv(
        args.output_dir / "segment_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [
            prediction_frame(
                "track",
                validation_tracks,
                validation_tracks["score"].to_numpy(),
                track_threshold,
            ),
            prediction_frame(
                "track", test_tracks, test_tracks["score"].to_numpy(), track_threshold
            ),
        ],
        ignore_index=True,
    ).to_csv(
        args.output_dir / "track_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": "SmallLogMelCNN",
        "input_shape": [1, 128, 249],
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "validation_segment_threshold": segment_threshold,
        "validation_track_threshold": track_threshold,
        "seed": SEED,
    }
    torch.save(checkpoint, args.output_dir / "small_logmel_cnn.pt")
    config = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "logmel_cache": str(args.logmel_cache),
        "codec_condition": cache_metadata["codec_condition"],
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "train_sampling": "50% REAL, 50% FAKE; FAKE provider-balanced",
        "frequency_mask_max_bands": 12,
        "time_mask_max_bins": 25,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 생성기별 Test 곡 성능: 같은 REAL 44곡과 각 생성기의 FAKE를 비교한다.
    real_tracks = test_tracks.loc[test_tracks["label_id"].eq(0)]
    generator_rows: list[dict[str, object]] = []
    for generator in sorted(test_tracks.loc[test_tracks["label_id"].eq(1), "generator"].unique()):
        subset = pd.concat(
            [
                real_tracks,
                test_tracks.loc[
                    test_tracks["label_id"].eq(1)
                    & test_tracks["generator"].eq(generator)
                ],
            ],
            ignore_index=True,
        )
        row = metric_row(
            "track",
            "test",
            subset["label_id"].to_numpy(),
            subset["score"].to_numpy(),
            track_threshold,
        )
        generator_rows.append(
            {
                "generator": generator,
                "fake_tracks": int(subset["label_id"].eq(1).sum()),
                "eer": row["eer"],
                "roc_auc": row["roc_auc"],
                "real_fpr": row["real_fpr"],
                "fake_miss_rate": row["fake_miss_rate"],
            }
        )
    generator_metrics = pd.DataFrame(generator_rows).sort_values("eer")
    generator_metrics.to_csv(
        args.output_dir / "generator_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 학습 곡선
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="Train")
    axes[0].plot(
        history_frame["epoch"], history_frame["validation_loss"], marker="o", label="Validation"
    )
    axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}")
    axes[0].set(title="CNN loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        history_frame["epoch"],
        history_frame["validation_segment_eer"] * 100,
        marker="o",
        label="Segment EER",
    )
    axes[1].plot(
        history_frame["epoch"],
        history_frame["validation_track_eer"] * 100,
        marker="o",
        label="Track EER",
    )
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch {best_epoch}")
    axes[1].set(title="Validation EER", xlabel="Epoch", ylabel="EER (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "learning_curves.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    # CNN Test 혼동행렬
    test_track_predictions = (
        test_tracks["score"].to_numpy() >= track_threshold
    ).astype(int)
    matrix = confusion_matrix(
        test_tracks["label_id"].to_numpy(), test_track_predictions, labels=[0, 1]
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set(
        title="Log-Mel CNN Test track confusion matrix",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
    )
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "test_track_confusion_matrix.png", dpi=170, bbox_inches="tight"
    )
    plt.close(figure)

    # 통제 조건 ML과 CNN 비교
    ml_metrics = pd.read_csv(args.ml_metrics)
    ml_test_track = ml_metrics.loc[
        ml_metrics["split"].eq("test") & ml_metrics["level"].eq("track")
    ][["model", "eer", "roc_auc", "macro_f1", "real_fpr", "fake_miss_rate"]]
    cnn_test_track = metrics.loc[
        metrics["split"].eq("test") & metrics["level"].eq("track")
    ][["model", "eer", "roc_auc", "macro_f1", "real_fpr", "fake_miss_rate"]]
    comparison = pd.concat([ml_test_track, cnn_test_track], ignore_index=True)
    comparison.to_csv(
        args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#4C78A8", "#E45756", "#72B7B2"]
    axes[0].bar(comparison["model"], comparison["eer"] * 100, color=colors)
    axes[1].bar(comparison["model"], comparison["roc_auc"] * 100, color=colors)
    for axis, column, title, ylabel in [
        (axes[0], "eer", "Test Track EER", "EER (%) - lower is better"),
        (axes[1], "roc_auc", "Test Track ROC-AUC", "ROC-AUC (%) - higher is better"),
    ]:
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f%%", padding=3)
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "ml_vs_cnn.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    display = comparison.copy()
    for column in ["eer", "roc_auc", "macro_f1", "real_fpr", "fake_miss_rate"]:
        display[column] = (display[column] * 100).round(2).astype(str) + "%"
    display.columns = [
        "모델",
        "EER",
        "ROC-AUC",
        "Macro-F1",
        "REAL 오탐률",
        "FAKE 놓침률",
    ]
    cnn_test = metrics.loc[
        metrics["split"].eq("test") & metrics["level"].eq("track")
    ].iloc[0]
    hardest = generator_metrics.sort_values("eer", ascending=False).iloc[0]
    easiest = generator_metrics.sort_values("eer").iloc[0]
    report = f"""# 7단계 Log-Mel CNN 결과

## 모델과 입력

- 입력: 공통 MP3 64kbps 조건의 10초 Log-Mel
- 입력 크기: 1 × 128 mel bands × 249 time bins
- CNN 파라미터: {parameter_count:,}개
- 장치: {device}
- Train sampling: REAL 50%, FAKE 50%; FAKE는 12개 생성기 균형
- 선택 기준: Validation Track EER
- 최적 epoch: {best_epoch}

128-band Log-Mel은 원래 약 996개 시간 frame을 가지지만, CPU·메모리 사용을
줄이기 위해 네 frame씩 평균하여 249개로 만들었다. 주파수축 128개는 유지했다.

## 통제 조건 Test 곡 단위 모델 비교

{markdown_table(display)}

- CNN Track EER: **{cnn_test['eer'] * 100:.2f}%**
- CNN Track ROC-AUC: **{cnn_test['roc_auc'] * 100:.2f}%**
- CNN Macro-F1: **{cnn_test['macro_f1'] * 100:.2f}%**

## 생성기별 관찰

- EER이 가장 낮은 생성기: **{easiest['generator']} ({easiest['eer'] * 100:.2f}%)**
- EER이 가장 높은 생성기: **{hardest['generator']} ({hardest['eer'] * 100:.2f}%)**

생성기별 결과는 `generator_metrics.csv`에 저장했다. 이 모델은 모든 12개
생성기를 학습에 포함한 표준 모델이므로, 생성기별 수치는 seen 성능이다.

## 해석 시 주의

이번 결과는 seed 42 한 번의 초기 CNN 기준선이다. 최종 보고 전에는 여러 seed로
반복해 평균과 변동을 제시해야 한다. 또한 MusicGen·Suno를 완전히 제외한 CNN을
따로 학습해야 미지 생성기 일반화를 ML 모델과 공정하게 비교할 수 있다.
"""
    (args.output_dir / "CNN_결과요약.md").write_text(report, encoding="utf-8")

    elapsed = time.monotonic() - started
    print("Log-Mel CNN 학습과 평가가 완료되었습니다.")
    print(
        f"best_epoch={best_epoch}, Test Track EER={cnn_test['eer']:.4f}, "
        f"ROC-AUC={cnn_test['roc_auc']:.4f}, elapsed={elapsed:.1f}s"
    )
    print(f"결과 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
