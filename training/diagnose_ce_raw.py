import torch
import torch.nn as nn
import random
import numpy as np
import sys

sys.path.insert(0, '.')

from models.bone_md_teacher_contrastive import BoneMDTeacherContrastive
from dataset.cached_multimodal_dataset import CachedMultimodalDataset
from torch.utils.data import DataLoader


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda')

print(f'Device: {device}')
print('Loading datasets...')


# --------------------------------------------------
# Dataset
# --------------------------------------------------
train_dataset = CachedMultimodalDataset(
    'cache/train'
)

val_dataset = CachedMultimodalDataset(
    'cache/val',
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print(f'Train samples: {len(train_dataset)}')
print(f'Val samples: {len(val_dataset)}')


# --------------------------------------------------
# Model
# --------------------------------------------------
model = BoneMDTeacherContrastive().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0
)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Inspect cached data
# --------------------------------------------------
batch = next(iter(train_loader))

print()
print('=== CACHE DATA STATS ===')

print(
    f"CT stats: "
    f"min={batch['ct'].min():.1f}, "
    f"max={batch['ct'].max():.1f}, "
    f"mean={batch['ct'].mean():.1f}, "
    f"std={batch['ct'].std():.1f}"
)

print(
    f"AP stats: "
    f"min={batch['ap'].min():.4f}, "
    f"max={batch['ap'].max():.4f}, "
    f"mean={batch['ap'].mean():.4f}"
)

print(
    f"Lateral stats: "
    f"min={batch['lateral'].min():.4f}, "
    f"max={batch['lateral'].max():.4f}, "
    f"mean={batch['lateral'].mean():.4f}"
)

print(
    f"Labels in first batch: "
    f"{batch['label'].tolist()}"
)


# --------------------------------------------------
# CE-only training
# --------------------------------------------------
print()
print('=== CE ONLY, LR=3e-4, RAW CT DATA ===')

for epoch in range(1, 16):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        ap = batch['ap'].to(device)
        lateral = batch['lateral'].to(device)
        ct = batch['ct'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(ap, lateral, ct)

        loss = criterion(
            output['logits'],
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        preds = output['logits'].argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total

    train_acc = (
        train_correct /
        train_total
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            ap = batch['ap'].to(device)
            lateral = batch['lateral'].to(device)
            ct = batch['ct'].to(device)
            labels = batch['label'].to(device)

            output = model(
                ap,
                lateral,
                ct
            )

            preds = output['logits'].argmax(dim=1)

            val_correct += (
                preds == labels
            ).sum().item()

            val_total += labels.size(0)

    val_acc = (
        val_correct /
        val_total
    )

    print(
        f'Epoch {epoch:02d}: '
        f'CE={train_loss:.4f}, '
        f'TrainAcc={train_acc:.4f}, '
        f'ValAcc={val_acc:.4f}',
        flush=True
    )


print()
print('Diagnostic finished.')