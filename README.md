# EMResformer
This repo contains the code for our paper "Iterative Optimal Attention and Local Model for Single Image Deraining Line''
![image](https://github.com/ghfkahfk/EMResformer/blob/main/framenetwork.png)

### Dependencies
Please install following essential dependencies (see requirements.txt):
```

torch==1.10.1
torchvision==0.11.2
cudatoolkit=10.2
matplotlib
scikit-learn
scikit-image
opencv-python
yacs
joblib
natsort
h5py
tqdm
einops
gdown
addict
future
lmdb
numpy
pyyaml
requests
scipy
tb-nightly
yapf
lpips
```

### Datasets
Download:  
Download:  
1. **Rain200H**  (https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)  
2. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](https://zmiclab.github.io/zxh/0/mscmrseg19)   


### Training  
python train.py

### Testing
python test.py



