# BoneMD-Net

## Multimodal Knowledge Distillation Network for Efficient Bone Disorder Diagnosis Using X-ray and CT Images

BoneMD-Net is a multimodal deep learning research project for three-class lumbar bone disorder classification using X-ray and CT imaging data from the LUMOS dataset.

### Classification Classes

- Class 0 — Normal
- Class 1 — Osteopenia
- Class 2 — Osteoporosis

---

## Final Model

The final verified model is:

**BoneMDTeacherNorm**

Architecture:

- AP X-ray encoder
- Lateral X-ray encoder
- CT encoder
- L2-normalized modality features
- Multimodal feature concatenation
- Fusion MLP
- Three-class classifier

Feature dimensions:

- AP X-ray: 512
- Lateral X-ray: 512
- CT: 512
- Fused representation: 512

The final checkpoint is:

checkpoints/FINAL_BEST_TEACHER_75_61.pth

Best validation accuracy:

**68.29%**

Test accuracy:

**75.61%**

Test set:

**41 patients**

Correct predictions:

**31 / 41**

---

## Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 75.61% |
| Macro Precision | 75.46% |
| Macro Recall | 77.81% |
| Macro F1 | 75.43% |
| Weighted F1 | 75.86% |

### Confusion Matrix

| Actual / Predicted | Normal | Osteopenia | Osteoporosis |
|---|---:|---:|---:|
| Normal | 13 | 4 | 2 |
| Osteopenia | 1 | 9 | 2 |
| Osteoporosis | 0 | 1 | 9 |

---

## Dataset

The project uses the LUMOS multimodal lumbar imaging dataset.

The processed experimental cohort contains:

**272 patients**

Patient-level split:

| Split | Patients |
|---|---:|
| Train | 190 |
| Validation | 41 |
| Test | 41 |

The processed cache is intentionally NOT included in this GitHub repository because it is approximately 21.5 GB.

The cache is required for reproducing the final inference/evaluation locally and should be transferred separately.

Expected cache structure:

`	ext
cache/
├── train/
│   └── *.pt
├── val/
│   └── *.pt
└── test/
    └── *.pt

