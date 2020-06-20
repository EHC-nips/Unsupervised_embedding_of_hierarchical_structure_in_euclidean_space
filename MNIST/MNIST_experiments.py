from model import *
import numpy as np
from util import *
#from vae import *

import argparse

import torch
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from torchvision import datasets, transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Valid argument for embedding_method: ward, complete, centroid, single, average
    parser.add_argument('-m', '--method', required=False, type=str, default='ward', help="number of clusters")
    parser.add_argument('-s', '--subsampling', required=False, type=int, default=100)
    parser.add_argument('-r', '--experiments_repeat', required=False, type=int, default=100)
    # Valid argument for embedding_method: VaDE, Origin, PCA, VAE
    parser.add_argument('-e', '--embedding_method', required = False, type=str, default = "VaDE")
    parser.add_argument('-t', '--rescaling_transform', action = "store_true")
    args = parser.parse_args()
    
    SUBSAMPLE_SIZE = args.subsampling
    repeat = args.experiments_repeat
    method = args.method
    emb = args.embedding_method
    dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor())
    mnist_data, cla = dataset.data.numpy().reshape(-1, 784) / 255, dataset.targets.numpy()

    #generate synthetic data
    vae = VAE()
    vae.load_state_dict(torch.load("pretrained_parameters/mnist_vae.pth", map_location=torch.device('cpu')))
    model = VaDE()
    model.load_state_dict(torch.load("pretrained_parameters/parameters_vade_linear_10classes_mnist.pth", map_location=torch.device('cpu')))
    # begin evaluation 
    print("DP:", compute_purity_average(model, mnist_data, cla, 10, 200, repeat, eval = emb, transform=args.rescaling_transform, VERBOSE = False, method = method))
    print("MW:", compute_MW_objective_average(model, mnist_data, cla, 10, 200, repeat, eval = emb,transform=args.rescaling_transform, VERBOSE = False, method = method))

    """
    print("VAE DP:", compute_purity_average(vae, mnist_data, cla, 10, 200, repeat, eval = "VAE", VERBOSE = True, method = method))
    print("VaDE transformed DP:", compute_purity_average(model, mnist_data, cla, 10, 200, repeat, eval = "VaDE", transform=True, VERBOSE = True, method = method))
    print("VaDE DP:", compute_purity_average(model, mnist_data, cla, 10, 200, repeat, eval = "VaDE", VERBOSE = True, method = method))
    print("PCA DP:", compute_purity_average(model, mnist_data, cla, 10, 200, repeat, eval = "PCA", VERBOSE = True, method = method))
    print("Origin DP:", compute_purity_average(model, mnist_data, cla, 10, 200, repeat, eval = "Origin", VERBOSE = True, method = method))
    
    
    print("VAE MW:", compute_MW_objective_average(vae, mnist_data, cla, 10, 200, repeat, eval = "VAE", VERBOSE = True, method = method))
    print("VaDE transformed MW:", compute_MW_objective_average(model, mnist_data, cla, 10, 200, repeat, eval = "VaDE",transform=True, VERBOSE = True, method = method))
    print("VaDE MW:", compute_MW_objective_average(model, mnist_data, cla, 10, 200, repeat, eval = "VaDE", VERBOSE = True, method = method))
    print("PCA MW:", compute_MW_objective_average(model, mnist_data, cla, 10, 200, repeat, eval = "PCA", VERBOSE = True, method = method))
    print("Origin MW:", compute_MW_objective_average(model, mnist_data, cla, 10, 200, repeat, eval = "Origin", VERBOSE = True, method = method))
    """
    
    