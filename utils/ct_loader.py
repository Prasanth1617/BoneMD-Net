import zipfile
from pathlib import Path

import numpy as np
import pydicom


def load_ct_volume(zip_path, patient_folder):
    """
    Load one patient's selected CT reconstruction directly from a LUMOS ZIP.

    Selection rules:
    1. Keep PRIMARY + AXIAL images only.
    2. Group by kernel + slice thickness + in-plane pixel spacing.
    3. Select the best reconstruction using:
       - smallest slice thickness
       - smallest in-plane pixel spacing
       - largest Z coverage
       - deterministic kernel name
    4. Sort slices by ImagePositionPatient[2].
    5. Remove duplicate Z positions.
    6. Convert raw pixels to Hounsfield Units (HU).

    Returns:
        volume: numpy array, shape [D, H, W], dtype float32
        metadata: list of dictionaries, one per retained slice
    """

    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:

        # ---------------------------------------------------------
        # 1. Find all DICOMs belonging to this patient
        # ---------------------------------------------------------
        names = [
            n
            for n in archive.namelist()
            if n.lower().endswith(".dcm")
            and "__macosx" not in n.lower()
            and n.startswith(patient_folder + "/")
        ]

        if not names:
            raise ValueError(
                f"No DICOM slices found for {patient_folder}"
            )

        # ---------------------------------------------------------
        # 2. Read metadata and keep PRIMARY + AXIAL only
        # ---------------------------------------------------------
        records = []

        for name in names:
            with archive.open(name) as f:
                ds = pydicom.dcmread(f, stop_before_pixels=True)

            image_type = [
                str(x).upper()
                for x in getattr(ds, "ImageType", [])
            ]

            if "PRIMARY" not in image_type:
                continue

            if "AXIAL" not in image_type:
                continue

            position = getattr(ds, "ImagePositionPatient", None)
            spacing = getattr(ds, "PixelSpacing", None)

            if position is None or spacing is None:
                continue

            kernel = str(
                getattr(ds, "ConvolutionKernel", "Unknown")
            )

            thickness = float(
                getattr(ds, "SliceThickness", 0.0)
            )

            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])
            z_position = float(position[2])

            records.append(
                {
                    "name": name,
                    "kernel": kernel,
                    "thickness": thickness,
                    "row_spacing": row_spacing,
                    "col_spacing": col_spacing,
                    "z": z_position,
                }
            )

        if not records:
            raise ValueError(
                f"No PRIMARY + AXIAL CT slices found for "
                f"{patient_folder}"
            )

        # ---------------------------------------------------------
        # 3. Group into reconstruction candidates
        # ---------------------------------------------------------
        candidates = {}

        for record in records:
            key = (
                record["kernel"],
                round(record["thickness"], 3),
                round(record["row_spacing"], 6),
                round(record["col_spacing"], 6),
            )

            candidates.setdefault(key, []).append(record)

        candidate_list = []

        for key, group in candidates.items():

            z_values = sorted(
                set(round(r["z"], 3) for r in group)
            )

            candidate_list.append(
                {
                    "key": key,
                    "records": group,
                    "slice_count": len(group),
                    "unique_z": len(z_values),
                    "z_min": min(z_values),
                    "z_max": max(z_values),
                    "coverage": max(z_values) - min(z_values),
                }
            )

        # ---------------------------------------------------------
        # 4. Select reconstruction
        # ---------------------------------------------------------
        #
        # Priority:
        #   smallest thickness
        #   smallest in-plane spacing
        #   largest coverage
        #   alphabetical kernel
        #
        selected = sorted(
            candidate_list,
            key=lambda c: (
                c["key"][1],
                c["key"][2],
                c["key"][3],
                -c["coverage"],
                c["key"][0],
            ),
        )[0]

        selected_records = selected["records"]

        # ---------------------------------------------------------
        # 5. Sort spatially by Z
        # ---------------------------------------------------------
        selected_records = sorted(
            selected_records,
            key=lambda r: r["z"]
        )

        # ---------------------------------------------------------
        # 6. Remove duplicate Z positions
        # ---------------------------------------------------------
        unique_records = []
        seen_z = set()

        for record in selected_records:

            z_key = round(record["z"], 3)

            if z_key in seen_z:
                continue

            seen_z.add(z_key)
            unique_records.append(record)

        # ---------------------------------------------------------
        # 7. Load pixels and convert to HU
        # ---------------------------------------------------------
        slices = []
        metadata = []

        for record in unique_records:

            with archive.open(record["name"]) as f:
                ds = pydicom.dcmread(f)

            pixel_array = ds.pixel_array.astype(np.float32)

            slope = float(
                getattr(ds, "RescaleSlope", 1.0)
            )

            intercept = float(
                getattr(ds, "RescaleIntercept", 0.0)
            )

            hu = pixel_array * slope + intercept

            # Normalize verified LUMOS CT padding values to air.
            hu[hu == -2048.0] = -1024.0
            hu[hu == -3024.0] = -1024.0

            slices.append(hu)

            metadata.append(
                {
                    "file": record["name"],
                    "z": record["z"],
                    "kernel": record["kernel"],
                    "thickness": record["thickness"],
                    "row_spacing": record["row_spacing"],
                    "col_spacing": record["col_spacing"],
                }
            )

        volume = np.stack(slices, axis=0).astype(
            np.float32
        )

        return volume, metadata


if __name__ == "__main__":

    zip_path = "dataset/lumos_ct_001_070_dcm.zip"
    patient_folder = "lumos_ct_001"

    volume, metadata = load_ct_volume(
        zip_path,
        patient_folder,
    )

    print()
    print("CT volume loaded successfully")
    print("--------------------------------")
    print("Shape:", volume.shape)
    print("dtype:", volume.dtype)
    print("HU minimum:", float(volume.min()))
    print("HU maximum:", float(volume.max()))
    print("HU mean:", float(volume.mean()))
    print("Retained slices:", len(metadata))
    print("Selected kernel:", metadata[0]["kernel"])
    print("Slice thickness:", metadata[0]["thickness"])
    print("First slice:", metadata[0]["file"])
    print("Last slice:", metadata[-1]["file"])