# Attention GAN
This repository provides GAN with various attention modules. 
All attention modules are applied to 32x32 feature map in both generator and discriminator(SA-GAN paper demonstrated when attention module apply to 32x32 feature map have the best fid result). 
Used wgan-hinge loss, spectral normalization and Adam optimizer for train, for discriminator and generator have different learning rate(4e-4 for discriminator, 1e-4 for generator) to satisfy TTUR(Two Time-scale Update Rule). 
Because we only use one class dataset(CelebA), replaced Conditional Batch Normalization with Batch Normalization.

## Provided Attention Module
### SA(Self Attention)  
+ ["Self-Attention Generative Adversarial Networks."](http://export.arxiv.org/pdf/1805.08318)  
  
![Self-Attention](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/self-attention.jpg)  
### CCA(Criss Cross Attention)  
+ ["CCNet: Criss-Cross Attention for Semantic Segmentation."](https://arxiv.org/pdf/1811.11721.pdf)  
  
![Criss-Cross-Attention](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/criss-cross-attention.jpg)  
### YLA(Your Local Attention)  
+ ["Your Local GAN: Designing Two Dimensional Local Attention Mechanisms for Generative Models."](https://arxiv.org/pdf/1911.12287)  
  
![Your-Local-Attention](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/your-local-attention.png)  
### ESA for YLA  
![ESA](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/ESA.png)  



## Number of Attention Head
| Attention Module     |  Num of Head   |
|:------------:|:-----------------:|
|Self Attention (SA)     | 1    |
|Criss Cross Attention (CCA)     | 2   |
|Your Local Attention (YLA)     | 8   |

## Results (Trained for 100K)
| Attention Module     |  FID    |
|:------------:|:-----------------:|
|Self Attention (SA)     | 32.516    |
|Criss Cross Attention (CCA)     | 30.016    |
|Your Local Attention (YLA)     | 32.691    |

## Generated Images (Trained for 100K)
### Generated from SA(Self Attention)-GAN  
![Generated img from SA-GAN](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/SA_generated.PNG)  
### Generated from CCA(Criss Cross Attention)-GAN  
![Generated img from CCA-GAN](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/CCA_generated.PNG)  
### Generated from YLA(Your Local Attention)-GAN  
![Generated img from YLA-GAN](https://github.com/dwro0121/Attention_GAN/blob/main/imgs/YLA_generated.PNG)  


## Requirements
>
* numpy>=1.20.3
* tqdm>=4.61.0
* pillow>=7.1.2
* torch>=1.7.1
* torchvision>=0.8.2




## Train

### run with default config  
```python train.py```  
### run with CCA example  
```python train.py --batch_size 64 --dataset CelebA --version CCA```

## Generate Images

### run with default config  
```python generate.py --load 100```  
### run with CCA example  
```python generate.py --version CCA --load 100```



## Adopted from

+ [Spetral Noramlization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
+ [FID Calculate](https://github.com/mseitzer/pytorch-fid)  
