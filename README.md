
# Unsupervised Embedding of Hierarchical Structure in Euclidean Space

This repository is the official implementation of [Unsupervised Embedding of Hierarchical Structure in Euclidean Space](https://arxiv.org). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Pre-trained Models

You can download pretrained models here:

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

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
