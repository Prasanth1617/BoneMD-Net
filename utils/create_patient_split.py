import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT = Path("results/final_multimodal_manifest.csv")
OUTPUT = Path("results/patient_split.csv")

RANDOM_SEED = 42

print("BoneMD-Net Patient-Level Dataset Split")
print("=" * 45)

# Load final verified manifest
df = pd.read_csv(INPUT)

print(f"Input patients: {len(df)}")
print(f"Input labels: {df['label'].value_counts().sort_index().to_dict()}")

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_SEED,
)

# Second split: split temporary equally -> 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_SEED,
)

# Assign split names
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# Combine
split_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True,
)

# Sort by patient ID for easier inspection
split_df = split_df.sort_values("patient_id").reset_index(drop=True)

# Validation checks
assert len(split_df) == len(df)
assert split_df["patient_id"].nunique() == len(df)
assert set(split_df["patient_id"]) == set(df["patient_id"])

split_counts = split_df["split"].value_counts()
assert set(split_counts.index) == {"train", "val", "test"}

# Check patient leakage
train_ids = set(split_df.loc[split_df["split"] == "train", "patient_id"])
val_ids = set(split_df.loc[split_df["split"] == "val", "patient_id"])
test_ids = set(split_df.loc[split_df["split"] == "test", "patient_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

# Check all three labels occur in every split
for split_name in ["train", "val", "test"]:
    labels = set(
        split_df.loc[split_df["split"] == split_name, "label"]
    )
    assert labels == {0, 1, 2}

# Save
split_df.to_csv(OUTPUT, index=False)

print("\nSPLIT SUMMARY")
print("-" * 45)

for split_name in ["train", "val", "test"]:
    subset = split_df[split_df["split"] == split_name]

    print(
        f"{split_name:5s}: {len(subset):3d} patients | "
        f"labels {subset['label'].value_counts().sort_index().to_dict()}"
    )

print("\nLEAKAGE CHECK")
print("-" * 45)
print(f"Train ∩ Val : {len(train_ids & val_ids)}")
print(f"Train ∩ Test: {len(train_ids & test_ids)}")
print(f"Val ∩ Test  : {len(val_ids & test_ids)}")

print("\nTotal patients:", len(split_df))
print("Saved:", OUTPUT)
print("\nSUCCESS")
