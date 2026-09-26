import zipfile
import pydicom
import numpy as np

tests = [
    ("dataset/lumos_ct_001_070_dcm.zip", "lumos_ct_001"),
    ("dataset/lumos_ct_071_140_dcm.zip", "lumos_ct_071"),
    ("dataset/lumos_ct_141_210_dcm.zip", "lumos_ct_141"),
    ("dataset/lumos_ct_211_280_dcm.zip", "lumos_ct_211"),
]

for zip_path, patient_folder in tests:
    print(f"\nPatient: {patient_folder}")

    with zipfile.ZipFile(zip_path) as z:
        names = [
            n for n in z.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and f"{patient_folder}/" in n
        ]

        rows = []

        for name in names:
            with z.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            z_pos = getattr(ds, "ImagePositionPatient", [None, None, None])[2]

            rows.append({
                "filename": name.split("/")[-1],
                "instance": getattr(ds, "InstanceNumber", None),
                "z": z_pos,
            })

    z_values = [r["z"] for r in rows if r["z"] is not None]

    print("Slices:", len(rows))
    print("Missing Z:", len(rows) - len(z_values))
    print("Unique InstanceNumbers:", len(set(r["instance"] for r in rows)), "/", len(rows))

    if len(z_values) > 1:
        sorted_z = sorted(z_values)
        differences = np.diff(sorted_z)

        print("Z range:", sorted_z[0], "to", sorted_z[-1])
        print("Unique Z spacings:", np.unique(np.round(differences, 4)))

    rows_sorted = sorted(rows, key=lambda r: r["z"])

    print("First 3 spatially sorted:")
    for r in rows_sorted[:3]:
        print(" ", r)

    print("Last 3 spatially sorted:")
    for r in rows_sorted[-3:]:
        print(" ", r)
