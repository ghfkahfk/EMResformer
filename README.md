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
2. **Rain200L** (https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
3. **Rain1400** (https://xueyangfu.github.io/projects/cvpr2017.html)
4. **Rain1200** (https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view?usp=sharing)
5. **Rain12** (https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md)


### Training  
python train.py

### Testing
python test.py



