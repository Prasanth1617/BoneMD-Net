import zipfile
import collections
import pydicom

ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    files = [
        n for n in z.namelist()
        if n.lower().endswith((".dcm", ".dicom")) and not n.startswith("__MACOSX/")
    ]

    photo = collections.Counter()
    bits = collections.Counter()
    shapes = collections.Counter()
    mods = collections.Counter()
    errors = []

    for n in files:
        try:
            with z.open(n) as f:
                ds = pydicom.dcmread(f)

            photo[str(getattr(ds, "PhotometricInterpretation", None))] += 1
            bits[int(getattr(ds, "BitsStored", -1))] += 1
            shapes[
                (
                    int(getattr(ds, "Rows", -1)),
                    int(getattr(ds, "Columns", -1)),
                )
            ] += 1
            mods[str(getattr(ds, "Modality", None))] += 1

        except Exception as e:
            errors.append((n, str(e)))

print("DICOM files:", len(files))
print("Photometric:", dict(photo))
print("BitsStored:", dict(bits))
print("Shapes:", dict(shapes))
print("Modality:", dict(mods))
print("Read errors:", len(errors))

if errors:
    print("First errors:")
    for item in errors[:5]:
        print(item)
