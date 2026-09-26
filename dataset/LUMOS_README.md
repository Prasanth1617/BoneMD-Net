LUMOS is the first publicly available multimodal dataset for lumbar osteoporosis screening. It provides a comprehensive resource to advance AI-driven research in bone health assessment, particularly for automated osteoporosis classification and bone mineral density (BMD) prediction. It can also be used for medical image segmentation, medical image synthesis (e.g., X-ray-to-CT), and other potential applications.



Dataset Overview
Patients: 803
Lumbar X-ray images: 1,620 (anteroposterior or lateral views)
Lumbar CT scans: 280
All DICOM images have been anonymized.
In lumos_clinical_data:
DXA measurements: bmd, bmc,T-scores, Z-scores for vertebrae L1–L4...
Demographics: Age, gender, ethnicity, height, weight, BMI
Osteoporosis labels: 0 = Normal(T-score ≥ −1.0);1 = Osteopenia(−2.5 < T-score < −1.0);2 = Osteoporosis (T-score ≤ −2.5)

Repository Structure
LUMOS/
├── lumos_x/ # X-ray DICOM files (lumos_x_XXX/lumos_x_XXX_1.dcm,...)
├── lumos_ct/ # CT scan series (lumos_ct_XXX/lumos_ct_XXX_1.dcm, lumos_ct_XXX_2.dcm,...)
├── lumos_clinical_data.xlsx # Clinical metadata: demographics, DXA results, labels
└── README.md
(Correction: lumos_x_741_670_dcm.zip should be renamed as lumos_x_741_770_dcm.zip)



License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
Commercial use requires prior written permission from the ethics committee of the host institution.



Citation
If you use this dataset in your work, please cite the original publication:
Keyue Shi, Qianqian Shen, Zhaoming Ye, Liangjun Jiang, Jiajun Bu, and Haishuai Wang. 2025. LUMOS: A Lumbar Multimodal Osteoporosis Screening Dataset with X-ray and CT images. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3746027.3758282

@inproceedings{shi2025lumos,
title={LUMOS: A Lumbar Multimodal Osteoporosis Screening Dataset with X-ray and CT images},
author={Shi, Keyue and Shen, Qianqian and Ye, Zhaoming and Jiang, Liangjun and Bu, Jiajun and Wang, Haishuai},
booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
pages={13250--13257},
year={2025}
}

For more details, visit the official project page: https://keyueshi.github.io/LUMOS/