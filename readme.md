# G2KD

Code (pytorch) for ['G2KD: Gradual Geometry-guided Knowledge Distillation for Source Data-free Domain Adaptation']() on Digit, Office-31, Office-Home, VisDA-C.

### Preliminary

You need to download the [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) dataset,  modify the path of images in each '.txt' under the folder './data/'.

Concerning the **Digits** dsatasets, the code will automatically download three digit datasets (i.e., MNIST, USPS, and SVHN) in './digit/data/'.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.3
- pytorch ==1.6.0
- torchvision == 0.7.0

## Prepare pretrain model for VIT 
We choose R50-ViT-B_16 as our encoder.
```bash root transformerdepth
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz 
mkdir ./model/vit_checkpoint/imagenet21k 
mv R50+ViT-B_16.npz ./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

### Training and evaluation

1. First training model on the source data,  Office-31 dataset on Resnet is shown here.

> ~/anaconda3/bin/python g2kd_source.py --trte val --output g2kd2020r0/source/ --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --seed 2020

2. Then adapting source model to target domain, with only the unlabeled target data.

> ~/anaconda3/bin/python g2kd_target.py --cls_par 0.05 --da uda --dset office --gpu_id 0 --s 0 --t 1 --output_src g2kd2020r0/source/ --output g2kd2020r0/target_mix/ --seed 2020


### Results

![](./G2KD_Res/object/results/accuracy/result_office.png)

**The results of g2kd is display under the folder './G2KD_Res/object/results/accuracy/'.**

### Acknowledgement

The codes are based on [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT).

### Contact

- tntechlab@hotmail.com



