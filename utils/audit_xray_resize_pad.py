import pandas as pd
import numpy as np


MAPPING = "results/xray_image_mapping.csv"
TARGET_SIZE = 224

df = pd.read_csv(MAPPING)
df["aspect_ratio"] = df["columns"] / df["rows"]

scale = np.minimum(
    TARGET_SIZE / df["rows"],
    TARGET_SIZE / df["columns"]
)

df["new_rows"] = np.round(df["rows"] * scale).astype(int)
df["new_cols"] = np.round(df["columns"] * scale).astype(int)

df["content_fraction"] = (
    df["new_rows"] * df["new_cols"]
) / (TARGET_SIZE * TARGET_SIZE)

df["padding_fraction"] = 1 - df["content_fraction"]


print("X-ray resize/padding audit")
print("==========================")
print(f"Images: {len(df)}")
print(f"Target size: {TARGET_SIZE} x {TARGET_SIZE}")
print()

print("Padding fraction:")
print(f"Minimum: {df['padding_fraction'].min():.3f}")
print(f"Median:  {df['padding_fraction'].median():.3f}")
print(f"Mean:    {df['padding_fraction'].mean():.3f}")
print(f"Maximum: {df['padding_fraction'].max():.3f}")
print()

print("Content fraction:")
print(f"Minimum: {df['content_fraction'].min():.3f}")
print(f"Median:  {df['content_fraction'].median():.3f}")
print(f"Mean:    {df['content_fraction'].mean():.3f}")
print(f"Maximum: {df['content_fraction'].max():.3f}")
print()

print("Images with >50% padding:",
      int((df["padding_fraction"] > 0.50).sum()))

print("Images with >60% padding:",
      int((df["padding_fraction"] > 0.60).sum()))

print("Images with >70% padding:",
      int((df["padding_fraction"] > 0.70).sum()))

print()

worst = df.nlargest(10, "padding_fraction")

print("10 images with most padding:")
print(
    worst[
        [
            "patient_id",
            "view",
            "rows",
            "columns",
            "aspect_ratio",
            "new_rows",
            "new_cols",
            "padding_fraction",
        ]
    ].to_string(index=False)
)
