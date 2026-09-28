import zipfile
from pathlib import Path

PATIENTS = [5, 21, 6, 35, 12, 74]

ROOT = Path("TEST_SAMPLES_6")

XRAY_ZIP = Path(
    r"..\BoneMD-Net\dataset\lumos_x_001_280_dcm.zip"
)

CT_ZIP_1 = Path(
    r"..\BoneMD-Net\dataset\lumos_ct_001_070_dcm.zip"
)

CT_ZIP_2 = Path(
    r"..\BoneMD-Net\dataset\lumos_ct_071_140_dcm.zip"
)


def main():
    if ROOT.exists():
        print("Removing old TEST_SAMPLES_6...")
        import shutil
        shutil.rmtree(ROOT)

    ROOT.mkdir()

    print("Opening X-ray archive...")
    with zipfile.ZipFile(XRAY_ZIP, "r") as xray_zip:

        print("Opening CT archives...")
        with zipfile.ZipFile(CT_ZIP_1, "r") as ct1_zip:
            with zipfile.ZipFile(CT_ZIP_2, "r") as ct2_zip:

                for patient in PATIENTS:

                    folder = ROOT / f"patient_{patient:03d}"
                    folder.mkdir()

                    # -------------------------
                    # X-rays
                    # -------------------------

                    ap_name = (
                        f"lumos_x_{patient:03d}/"
                        f"lumos_x_{patient:03d}_1.Dcm"
                    )

                    lateral_name = (
                        f"lumos_x_{patient:03d}/"
                        f"lumos_x_{patient:03d}_2.Dcm"
                    )

                    ap_data = xray_zip.read(ap_name)
                    lateral_data = xray_zip.read(lateral_name)

                    (folder / "AP.dcm").write_bytes(ap_data)
                    (folder / "Lateral.dcm").write_bytes(lateral_data)

                    # -------------------------
                    # CT
                    # -------------------------

                    if patient <= 70:
                        ct_zip = ct1_zip
                    else:
                        ct_zip = ct2_zip

                    prefix = f"lumos_ct_{patient:03d}/"

                    ct_names = [
                        name
                        for name in ct_zip.namelist()
                        if name.startswith(prefix)
                        and name.lower().endswith(".dcm")
                        and "__macosx" not in name.lower()
                    ]

                    print(
                        f"Patient {patient:03d}: "
                        f"{len(ct_names)} CT slices"
                    )

                    if not ct_names:
                        raise RuntimeError(
                            f"No CT slices found for patient {patient:03d}"
                        )

                    ct_output = folder / "CT.zip"

                    with zipfile.ZipFile(
                        ct_output,
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as output_zip:

                        for name in ct_names:
                            filename = Path(name).name
                            output_zip.writestr(
                                filename,
                                ct_zip.read(name),
                            )

                    print(
                        f"Patient {patient:03d}: sample created"
                    )

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Created: {ROOT.resolve()}")


if __name__ == "__main__":
    main()