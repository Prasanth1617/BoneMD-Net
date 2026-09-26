import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "teacher_contrastive_queue_best.pth"
)

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


def collect_outputs(model, loader):

    model.eval()

    features = []
    labels = []
    logits_list = []

    with torch.no_grad():

        for batch in loader:

            ap = batch["ap"].to(DEVICE)
            lateral = batch["lateral"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)

            output = model(ap, lateral, ct)

            features.append(
                output["fused_features"].cpu()
            )

            logits_list.append(
                output["logits"].cpu()
            )

            labels.append(
                batch["label"].cpu()
            )

    return (
        torch.cat(features),
        torch.cat(logits_list),
        torch.cat(labels),
    )


def find_classifier_layers(model):

    layers = []

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            layers.append(
                (name, module)
            )

    return layers


def main():

    print("Device:", DEVICE)

    model = BoneMDTeacherContrastive(
        feature_dim=512,
        num_classes=3,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

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

    print()

    # ------------------------------------------------------------
    # Identify classifier layers
    # ------------------------------------------------------------

    print("=" * 75)
    print("LINEAR LAYERS")
    print("=" * 75)

    linear_layers = find_classifier_layers(model)

    for name, layer in linear_layers:

        print(
            f"{name}: "
            f"in_features={layer.in_features}, "
            f"out_features={layer.out_features}"
        )

    print()

    # The final Linear layer is the 3-class classifier.
    final_name = "classifier.3"
    classifier = dict(linear_layers)[final_name]

    print(
        "Final classifier:",
        final_name,
    )

    print(
        "Weight shape:",
        tuple(classifier.weight.shape),
    )

    print(
        "Bias shape:",
        tuple(classifier.bias.shape),
    )

    print()

    # ------------------------------------------------------------
    # Classifier weight geometry
    # ------------------------------------------------------------

    weights = classifier.weight.detach().cpu()
    bias = classifier.bias.detach().cpu()

    print("=" * 75)
    print("CLASSIFIER WEIGHT GEOMETRY")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        norm = torch.linalg.vector_norm(
            weights[class_id]
        ).item()

        print(
            f"{CLASS_NAMES[class_id]} "
            f"weight norm: {norm:.6f}"
        )

    print()

    print("CLASSIFIER WEIGHT DISTANCES")
    print("-" * 75)

    for a, b in [
        (0, 1),
        (0, 2),
        (1, 2),
    ]:

        distance = torch.linalg.vector_norm(
            weights[a] - weights[b]
        ).item()

        cosine = F.cosine_similarity(
            weights[a].unsqueeze(0),
            weights[b].unsqueeze(0),
        ).item()

        print(
            f"{CLASS_NAMES[a]} <-> "
            f"{CLASS_NAMES[b]}: "
            f"distance={distance:.6f}, "
            f"cosine={cosine:.6f}"
        )

    print()

    print("CLASSIFIER BIASES")
    print("-" * 75)

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]}: "
            f"{bias[class_id].item():.6f}"
        )

    print()

    # ------------------------------------------------------------
    # Train / validation outputs
    # ------------------------------------------------------------

    train_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "train"
    )

    val_dataset = CachedMultimodalDataset(
        PROJECT_ROOT / "cache" / "val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    train_features, train_logits, train_labels = (
        collect_outputs(
            model,
            train_loader,
        )
    )

    val_features, val_logits, val_labels = (
        collect_outputs(
            model,
            val_loader,
        )
    )

    # ------------------------------------------------------------
    # Logit statistics
    # ------------------------------------------------------------

    print("=" * 75)
    print("TRAIN LOGIT STATISTICS")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        mask = train_labels == class_id

        class_logits = train_logits[mask]

        print()
        print(
            f"True class: {CLASS_NAMES[class_id]}"
        )

        for predicted_class in [0, 1, 2]:

            values = class_logits[
                :,
                predicted_class,
            ]

            print(
                f"  {CLASS_NAMES[predicted_class]} "
                f"logit: mean={values.mean().item():.6f}, "
                f"std={values.std().item():.6f}"
            )

    print()

    print("=" * 75)
    print("VALIDATION LOGIT STATISTICS")
    print("=" * 75)

    for class_id in [0, 1, 2]:

        mask = val_labels == class_id

        class_logits = val_logits[mask]

        print()
        print(
            f"True class: {CLASS_NAMES[class_id]}"
        )

        for predicted_class in [0, 1, 2]:

            values = class_logits[
                :,
                predicted_class,
            ]

            print(
                f"  {CLASS_NAMES[predicted_class]} "
                f"logit: mean={values.mean().item():.6f}, "
                f"std={values.std().item():.6f}"
            )

    print()

    # ------------------------------------------------------------
    # Classification margins
    # ------------------------------------------------------------

    print("=" * 75)
    print("CLASSIFICATION MARGINS")
    print("=" * 75)

    def analyze_margins(
        logits,
        labels,
        split_name,
    ):

        predictions = logits.argmax(
            dim=1
        )

        margins = []

        for i in range(len(logits)):

            true_class = int(
                labels[i]
            )

            true_logit = logits[
                i,
                true_class,
            ]

            other_logits = torch.cat(
                [
                    logits[
                        i,
                        :true_class,
                    ],
                    logits[
                        i,
                        true_class + 1:
                    ],
                ]
            )

            strongest_other = other_logits.max()

            margin = (
                true_logit
                - strongest_other
            )

            margins.append(
                margin.item()
            )

        margins = torch.tensor(
            margins
        )

        correct_mask = (
            predictions == labels
        )

        incorrect_mask = (
            ~correct_mask
        )

        print()
        print(split_name)

        print(
            "Mean true-class margin:",
            margins.mean().item(),
        )

        if correct_mask.any():

            print(
                "Correct mean margin:",
                margins[
                    correct_mask
                ].mean().item(),
            )

        if incorrect_mask.any():

            print(
                "Incorrect mean margin:",
                margins[
                    incorrect_mask
                ].mean().item(),
            )

        for class_id in [0, 1, 2]:

            mask = labels == class_id

            if not mask.any():
                continue

            print(
                f"{CLASS_NAMES[class_id]} "
                f"mean margin:",
                margins[mask].mean().item(),
            )

    analyze_margins(
        train_logits,
        train_labels,
        "TRAIN",
    )

    analyze_margins(
        val_logits,
        val_labels,
        "VALIDATION",
    )

    print()

    # ------------------------------------------------------------
    # Classifier hidden representation versus classifier weights
    # ------------------------------------------------------------

    print("=" * 75)
    print("CLASSIFIER HIDDEN REPRESENTATION / WEIGHT ALIGNMENT")
    print("=" * 75)

    # classifier.0 maps fused 512-D features -> 256-D hidden features
    classifier_hidden = model.classifier[0]

    with torch.no_grad():

        train_hidden = F.relu(
            classifier_hidden(
                train_features.to(DEVICE)
            )
        ).cpu()

        val_hidden = F.relu(
            classifier_hidden(
                val_features.to(DEVICE)
            )
        ).cpu()

    print(
        "Train hidden shape:",
        tuple(train_hidden.shape),
    )

    print(
        "Val hidden shape:",
        tuple(val_hidden.shape),
    )

    # ------------------------------------------------------------
    # Hidden-space class centroids
    # ------------------------------------------------------------

    hidden_centroids = []

    for class_id in [0, 1, 2]:

        mask = train_labels == class_id

        centroid = train_hidden[
            mask
        ].mean(dim=0)

        hidden_centroids.append(
            centroid
        )

        print(
            f"{CLASS_NAMES[class_id]} "
            f"hidden centroid norm:",
            torch.linalg.vector_norm(
                centroid
            ).item(),
        )

    hidden_centroids = torch.stack(
        hidden_centroids
    )

    print()

    # ------------------------------------------------------------
    # Classifier weight alignment
    # ------------------------------------------------------------

    normalized_centroids = F.normalize(
        hidden_centroids,
        dim=1,
    )

    normalized_weights = F.normalize(
        weights,
        dim=1,
    )

    alignment = (
        normalized_centroids
        @ normalized_weights.T
    )

    print(
        "Cosine alignment:"
    )

    print(
        "Rows = true class centroids"
    )

    print(
        "Columns = classifier weight vectors"
    )

    print()

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]} centroid:"
        )

        for classifier_class in [0, 1, 2]:

            print(
                f"  vs "
                f"{CLASS_NAMES[classifier_class]}: "
                f"{alignment[class_id, classifier_class].item():.6f}"
            )

    print()

    # ------------------------------------------------------------
    # Correct classifier alignment
    # ------------------------------------------------------------

    print(
        "DIAGONAL VS OFF-DIAGONAL ALIGNMENT"
    )

    print("-" * 75)

    for class_id in [0, 1, 2]:

        diagonal = alignment[
            class_id,
            class_id,
        ].item()

        off_diagonal = torch.cat(
            [
                alignment[
                    class_id,
                    :class_id,
                ],
                alignment[
                    class_id,
                    class_id + 1:
                ],
            ]
        )

        strongest_wrong = (
            off_diagonal.max().item()
        )

        print(
            f"{CLASS_NAMES[class_id]}: "
            f"correct={diagonal:.6f}, "
            f"strongest_wrong={strongest_wrong:.6f}, "
            f"gap={diagonal - strongest_wrong:.6f}"
        )

    print()

    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()