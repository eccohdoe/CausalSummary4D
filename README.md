# (CVPR 2025) Mamba4D: Efficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models
## 📣 News
- **[8/April/2025]** We have released the main codes for human action recognition and 4D semantic segmentation.
- **[27/Feb/2025]** Our paper has been accepted by **CVPR 2025**! 🥳🥳🥳 
## Introduction
Point cloud videos can faithfully capture real-world spatial geometries and temporal dynamics, which are essential for enabling intelligent agents to understand the dynamically changing world. However, designing an effective 4D backbone remains challenging, mainly due to the irregular and unordered distribution of points and temporal inconsistencies across frames. Also, recent transformer-based 4D backbones commonly suffer from large computational costs due to their quadratic complexity, particularly for long video sequences. To address these challenges, we propose a novel point cloud video understanding backbone purely based on the State Space Models (SSMs). Specifically, we first disentangle space and time in 4D video sequences and then establish the spatio-temporal correlation with our designed Mamba blocks. The Intra-frame Spatial Mamba module is developed to encode locally similar geometric structures within a certain temporal stride. Subsequently, locally correlated tokens are delivered to the Inter-frame Temporal Mamba module, which integrates long-term point features across the entire video with linear complexity. Our proposed Mamba4d achieves competitive performance on the MSR-Action3D action recognition (+10.4% accuracy), HOI4D action segmentation (+0.7 F1 Score), and Synthia4D semantic segmentation (+0.19 mIoU) datasets. Especially, for long video sequences, our method has a significant efficiency improvement with 87.5% GPU memory reduction and 
5.36 speed-up.

## Installation
1. Mamba
    The mamba environment is installed by pulling the docker image.
    Create container and install the following libraries.
2. KNN_CUDA
    Follow [KNN_CUDA ](https://github.com/unlimblue/KNN_CUDA) to install.
3. Pointnet2-Pytorch
    Follow [Pointnet2 ](https://github.com/erikwijmans/Pointnet2_PyTorch) to install.

## Data Preprocessing
- The preprocess of the MSR-Action3D and Synthia4D datasets follows this work: [MeteorNet](https://github.com/xingyul/meteornet)

## Train
### 3D Action Recognition
- Train the model with intra-mamba
```sh
python3 train-msr.py  --data-path <Path to MSR-Action3D> --output_dir <Path to output>
```
- Train the model without intra-mamba
```sh
python3 train-msr.py --intra False --data-path <Path to MSR-Action3D> --output_dir <Path to output>
```

### 4D Semantic Segmentation
```sh
python3 train-syn.py \
    --data-path <Path to SYNTHIA>/sequence \
    --data-train <Path to SYNTHIA>/trainval_raw.txt \
    --data-eval <Path to SYNTHIA>/test_raw.txt \
    --output-dir <Path to output>
```

## Related Repos
- [P4Transformer](https://github.com/hehefan/P4Transformer)
- [Mamba](https://github.com/state-spaces/mamba)
- [PointMamba](https://github.com/LMD0311/PointMamba)

## Citation
```
@article{liu2024mamba4d,
  title={Mamba4d: Efficient long-sequence point cloud video understanding with disentangled spatial-temporal state space models},
  author={Liu, Jiuming and Han, Jinru and Liu, Lihao and Aviles-Rivero, Angelica I and Jiang, Chaokang and Liu, Zhe and Wang, Hesheng},
  journal={arXiv preprint arXiv:2405.14338},
  year={2024}
}
```
