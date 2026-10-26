# Third-party material

- **ResNet-50 + DeepLabV3+** (`segmentation/models.py`, class `DeepLabV3PlusResNet50`): the architecture is ported from the
  *HTSM Oil Spill Segmentation* repository (Satyanarayana and Dhali), which was the code of the published Caspian study. Its licence:

  > MIT License. Copyright (c) 2023 Abhishek Ramanathapura Satyanarayana. Permission is hereby granted, free of charge, to any person
  > obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction
  > [...]. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

- **SegFormer** weights pre-trained on ADE20K (`nvidia/segformer-b0-finetuned-ade-512-512`, `nvidia/segformer-b2-finetuned-ade-512-512`) are
  downloaded from the Hugging Face Hub at training time; the licence of those weights is the one given on their model cards.
  The two-scale design (HRDA) follows Hoyer et al., *HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation* (ECCV 2022);
  the code here is written from the description, not copied.
- **M4D-SAR-OilSpill** (Krestenitis et al., 2019) is not part of this repository; see the README.
