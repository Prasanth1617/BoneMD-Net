import zipfile
from pathlib import Path
from collections import Counter

import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "results" / "final_multimodal_manifest.csv"


def find_ct_dicom_files(zip_path, patient_folder):
    with zipfile.ZipFile(zip_path, "r") as z:
        return [
            name
            for name in z.namelist()
            if name.lower().endswith(".dcm")
            and "__macosx" not in name.lower()
            and name.startswith(patient_folder + "/")
        ]


def get_kernel(ds):
    return str(getattr(ds, "ConvolutionKernel", "")).strip() or "Unknown"


def get_image_type(ds):
    value = getattr(ds, "ImageType", [])
    return "\\".join(str(x) for x in value)


def main():
    manifest = pd.read_csv(MANIFEST)

    print("CT Kernel / Reconstruction Audit")
    print("================================")
    print(f"Eligible patients: {len(manifest)}")
    print()

    patient_rows = []
    global_kernels = Counter()

    for _, row in manifest.iterrows():
        patient_id = int(row["patient_id"])
        zip_path = PROJECT_ROOT / "dataset" / row["ct_zip"]
        patient_folder = row["ct_patient_folder"]

        names = find_ct_dicom_files(zip_path, patient_folder)

        kernels = Counter()
        image_types = Counter()
        series_descriptions = Counter()

        with zipfile.ZipFile(zip_path, "r") as z:
            for name in names:
                with z.open(name) as f:
                    ds = pydicom.dcmread(
                        f,
                        stop_before_pixels=True,
                        specific_tags=[
                            "ConvolutionKernel",
                            "ImageType",
                            "SeriesDescription",
                        ],
                    )

                kernel = get_kernel(ds)
                image_type = get_image_type(ds)
                series_description = str(
                    getattr(ds, "SeriesDescription", "")
                ).strip() or "Unknown"

                kernels[kernel] += 1
                image_types[image_type] += 1
                series_descriptions[series_description] += 1
                global_kernels[kernel] += 1

        patient_rows.append(
            {
                "patient_id": patient_id,
                "dicom_count": len(names),
                "kernel_count": len(kernels),
                "kernels": " | ".join(
                    f"{k}:{v}" for k, v in sorted(kernels.items())
                ),
                "image_types": " | ".join(
                    f"{k}:{v}" for k, v in sorted(image_types.items())
                ),
                "series_descriptions": " | ".join(
                    f"{k}:{v}" for k, v in sorted(series_descriptions.items())
                ),
            }
        )

    result = pd.DataFrame(patient_rows)

    output = PROJECT_ROOT / "results" / "ct_kernel_audit.csv"
    result.to_csv(output, index=False)

    print("Global kernel distribution")
    print("-------------------------")
    for kernel, count in global_kernels.most_common():
        print(f"{kernel}: {count}")

    print()
    print("Patients with multiple kernels")
    print("------------------------------")

    multi = result[result["kernel_count"] > 1]

    if len(multi) == 0:
        print("None")
    else:
        for _, row in multi.iterrows():
            print(
                f"Patient {int(row['patient_id']):03d}: "
                f"{row['kernels']}"
            )

    print()
    print(f"Patients with multiple kernels: {len(multi)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
