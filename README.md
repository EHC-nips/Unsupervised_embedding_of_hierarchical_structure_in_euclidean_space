
# Unsupervised Embedding of Hierarchical Structure in Euclidean Space

This repository is the official implementation of [Unsupervised Embedding of Hierarchical Structure in Euclidean Space](https://arxiv.org/abs/2010.16055). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Required Datasets will be downloaded automatically while running the code

## Training

To train and evaluate the model on synthetic data, run this command:

```
cd synthetic/
python synthetic_experiments.py --n_class 8 --margin 8 --variance 1 --dim 100 --hidden_dim 3 --linkage_method ward --learning_rate 1e-3
```

> The above command will give you the result for the BTGM in Figure 1 in our paper. 

## Pre-trained Models

We provide the pre-trained parameters for MNIST and CIFAR-25 (both in Pytorch) and reuters (from the [original implementation of VaDE](https://github.com/slim1017/VaDE)). You can download pretrained models here:

- Download parameters for CIFAR25 experiments to `CIFAR25/parameters/` using [this link](https://drive.google.com/file/d/1QljVdElZtRAM9b6kLqjCDUUeWETQ8u7a/view?usp=sharing) <br>
- Download reuters10k data to `reuters/dataset/reuters10k` using [this link](https://drive.google.com/file/d/13o7XuyqtzqJD8V7OcAZdIWfKo8GmZB-B/view?usp=sharing) <br>


## Evaluation

To evaluate on MNIST, run:

```
cd MNIST/
python MNIST_experiments.py --linkage_method ward --embedding_method VaDE --rescaling_transform
```

> 📋The same procedure applies to CIFAR-25 dataset.


## Results

Our proposed method achieves the following performance in terms of Dendrogram Purity and Moseley-Wang's objective :


| Model name         |      MNIST      |    CIFAR-25    |     reuters     |     20newsgroups   |
| ------------------ |---------------- | -------------- | --------------- | -------------------|
| VaDE+Ward+Trans.(DP)|      0.886      |     0.128      |      0.672      |      0.251        |
| VaDE+Ward+Trans.(MW)|      0.955      |     0.472      |      0.768      |      0.606        |

