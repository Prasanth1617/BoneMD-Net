import pandas as pd

df = pd.read_csv("results/xray_image_mapping.csv")
df["aspect_ratio"] = df["columns"] / df["rows"]

print("Images:", len(df))
print("Unique shapes:", df.groupby(["rows", "columns"]).ngroups)
print("Rows range:", df["rows"].min(), "-", df["rows"].max())
print("Columns range:", df["columns"].min(), "-", df["columns"].max())
print(
    "Aspect ratio range:",
    round(df["aspect_ratio"].min(), 3),
    "-",
    round(df["aspect_ratio"].max(), 3),
)

print()
print("Aspect-ratio summary:")
print(df["aspect_ratio"].describe().round(3))

print()
print("Most common shapes:")
print(
    df.groupby(["rows", "columns"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)
