import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bone_md_teacher_tscore import BoneMDTeacherTScore
from training.train_teacher_tscore import (
    TeacherTScoreDataset,
    TSCORE_COLUMNS,
    TSCORE_MEAN,
    TSCORE_STD,
    DEVICE,
    CLINICAL_FILE,
    TRAIN_CACHE,
    VAL_CACHE,
    CHECKPOINT_PATH,
)


def evaluate_split(model, dataset, split_name):

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    predictions = []
    targets = []
    patient_ids = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(
                ap,
                lateral,
                ct,
            )

            predicted = output["tscore_predictions"][0]
            predicted = predicted.cpu().numpy()

            target = batch["tscores"][0].numpy()

            # Only evaluate samples with complete T-score targets.
            if bool(batch["tscore_mask"][0].item()):

                predictions.append(predicted)
                targets.append(target)
                patient_ids.append(
                    int(batch["patient_id"][0])
                )

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Convert normalized predictions/targets back
    # to original T-score units.
    predictions_original = (
        predictions * TSCORE_STD
        + TSCORE_MEAN
    )

    targets_original = (
        targets * TSCORE_STD
        + TSCORE_MEAN
    )

    absolute_errors = np.abs(
        predictions_original
        - targets_original
    )

    mae_per_target = absolute_errors.mean(axis=0)

    overall_mae = absolute_errors.mean()

    print()
    print("=" * 60)
    print(f"{split_name.upper()} T-SCORE DIAGNOSTIC")
    print("=" * 60)

    print(
        "Samples with complete T-scores:",
        len(patient_ids),
    )

    print()
    print("MAE PER T-SCORE:")
    
    for column, mae in zip(
        TSCORE_COLUMNS,
        mae_per_target,
    ):
        print(
            f"{column:10s}: {mae:.4f}"
        )

    print()
    print(
        f"Overall T-score MAE: {overall_mae:.4f}"
    )

    print()
    print("Prediction statistics:")

    for i, column in enumerate(TSCORE_COLUMNS):

        print(
            f"{column:10s} "
            f"target_mean={targets_original[:, i].mean():.4f} "
            f"pred_mean={predictions_original[:, i].mean():.4f} "
            f"target_std={targets_original[:, i].std():.4f} "
            f"pred_std={predictions_original[:, i].std():.4f}"
        )

    return {
        "predictions": predictions_original,
        "targets": targets_original,
        "patient_ids": patient_ids,
    }


def main():

    print("Device:", DEVICE)

    clinical_df = pd.read_excel(
        CLINICAL_FILE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = BoneMDTeacherTScore(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint["val_accuracy"],
    )

    train_dataset = TeacherTScoreDataset(
        TRAIN_CACHE,
        clinical_df,
    )

    val_dataset = TeacherTScoreDataset(
        VAL_CACHE,
        clinical_df,
    )

    train_result = evaluate_split(
        model,
        train_dataset,
        "train",
    )

    val_result = evaluate_split(
        model,
        val_dataset,
        "validation",
    )

    print()
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()