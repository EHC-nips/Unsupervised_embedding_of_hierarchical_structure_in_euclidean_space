
Code for the paper "Unsupervised Embedding of Hierarchical Structure in Euclidean Space" <br>

All the results in the paper can be achieved by running
`python NAME_OF_DATASET_experiments.py`
<br>
# Unsupervised Embedding of Hierarchical Structure in Euclidean Space

This repository is the official implementation of [Unsupervised Embedding of Hierarchical Structure in Euclidean Space](https://arxiv.org). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- Download parameters for CIFAR100 experiments to `CIFAR100/parameters/` using [this link](https://drive.google.com/file/d/1QljVdElZtRAM9b6kLqjCDUUeWETQ8u7a/view?usp=sharing) <br>
- Download reuters10k data to `reuters/dataset/reuters10k` using [this link](https://drive.google.com/file/d/13o7XuyqtzqJD8V7OcAZdIWfKo8GmZB-B/view?usp=sharing) <br>

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
