from cifar100_util import *
from cifar100_model import *
#from vae import VAE

import numpy as np

import argparse

import torch
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from torchvision import datasets, transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--method', required=False, type=str, default='ward', help="number of clusters")
    parser.add_argument('-s', '--subsampling', required=False, type=int, default=100)
    args = parser.parse_args()
    
    SUBSAMPLE_SIZE = args.subsampling
    method = args.method
    
    X,Y = prepare_cifar100()

    #generate synthetic data
    
    vae = VAE()
    vae.load_state_dict(torch.load("parameters/cifar100_vae.pth", map_location=torch.device('cpu')))
    
    model = VaDE(3*32*32, 20, 25)
    model.load_state_dict(torch.load("parameters/parameters_vade_linear_25classes_cifar100_subset.pth", map_location=torch.device('cpu')))
    
    # begin evaluation 
    print("VAE DP:", compute_purity_average(vae, X, Y, 5, 50, eval = "VAE", method = method, VERBOSE = True))
    print("VaDE transformed DP:", compute_purity_average(model, X, Y, 5, 50, eval = "VaDE", transform=True, method = method, VERBOSE = True))
    print("VaDE DP:", compute_purity_average(model, X, Y, 5, 50, eval = "VaDE", method = method, VERBOSE = True))
    print("PCA DP:", compute_purity_average(model, X, Y, 5, 50, eval = "PCA", method = method, VERBOSE = True))
    print("Origin DP:", compute_purity_average(model, X,Y, 5, 50, eval = "Origin", method = method, VERBOSE = True))
    
    '''
    print("VAE MW:", compute_MW_objective_average(vae, X, Y, 25, 50, eval = "VAE", method = method, VERBOSE = True))
    print("VaDE transformed MW:", compute_MW_objective_average(model, X, Y, 25, 50, eval = "VaDE",transform=True, method = method, VERBOSE = True))
    print("VaDE MW:", compute_MW_objective_average(model, X, Y, 25, 50, eval = "VaDE", method = method, VERBOSE = True))
    print("PCA MW:", compute_MW_objective_average(model, X, Y, 25, 50, eval = "PCA", method = method, VERBOSE = True))
    print("Origin MW:", compute_MW_objective_average(model, X, Y, 25, 50, eval = "Origin", method = method, VERBOSE = True))
    '''
    
    
    
    
    methods_list = ["average", "centroid", "complete", "single", "ward"]
    for method in methods_list:
        print(method)
        print("VaDE transformed DP:", compute_purity_average(model, X, Y, 25, 50, eval = "VaDE", transform=True, method = method, VERBOSE = True, super_class = False))
        print("VaDE transformed MW:", compute_MW_objective_average(model, X, Y, 25, 50, eval = "VaDE",transform=True, method = method, VERBOSE = True))
        

    
    