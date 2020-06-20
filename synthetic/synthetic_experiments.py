from model import *
import numpy as np
from util import *

import argparse

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_class', required=False, type=int, default='8', help="number of clusters")
    parser.add_argument('-m', '--margin', required=False, type=float , default='4', help="margin of HGMM")
    parser.add_argument('-v', '--variance', required=False, type=float , default='1', help="variance of HGMM")
    parser.add_argument('-d', '--dim', required=False, type=int, default='100', help="dimension of HGMM")
    parser.add_argument('-hd', '--hidden_dim', required=False, type=int, default=10, help="hidden dimension size for VaDE model")
    parser.add_argument('-s', '--subsampling', required=False, type=int, default=100)
    parser.add_argument('-l', '--linkage_method', required=False, type=str, default="ward")
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=1e-3)
    args = parser.parse_args()
    
    
    N_CLASS = args.n_class
    MARGIN = args.margin
    VAR = args.variance
    DIM = args.dim
    HID_DIM = args.hidden_dim
    SUBSAMPLE_SIZE = args.subsampling
    lr = args.learning_rate
    N = 2000 # num per class


    #generate synthetic data
    train_loader, synthetic_data, cla = create_data_loader(400, N_CLASS,MARGIN,VAR,DIM,N)
    #train VAE
    vae = VAE(DIM, HID_DIM)
    #train_vae(vae, train_loader, 80)
    #torch.save(vae.state_dict(), "parameters/VAE_parameters_C{}_M{}.pth".format(args.n_class, args.margin))
    vae.load_state_dict(torch.load("parameters/VAE_parameters_C{}_M{}.pth".format(args.n_class, args.margin)))
    # train VaDE
    model = VaDE(N_CLASS, HID_DIM, DIM)
    model.pre_train(train_loader,pre_epoch=50)
    train(model, train_loader, 100, lr = lr)
    torch.save(model.state_dict(), "parameters/VaDE_parameters_C{}_M{}.pth".format(args.n_class, args.margin))
    #model.load_state_dict(torch.load("parameters/VaDE_parameters_C{}_M{}.pth".format(args.n_class, args.margin)))
    # begin evaluation 
    
    _, vae_mean, _ = vae(torch.from_numpy(synthetic_data).float())
    mean, _ = model.encoder(torch.from_numpy(synthetic_data).float())
    scaled_mean = transformation(model, synthetic_data)
    pca = PCA(n_components = HID_DIM)
    projection = pca.fit_transform(synthetic_data)
    print("VAE:", compute_purity_average(vae_mean.detach().numpy(), cla, N_CLASS, 2048, 50, method = args.linkage_method))
    #print("Transform:", compute_purity_average(scaled_mean.detach().numpy(), cla, N_CLASS, 2048, 50, method = args.linkage_method))
    print("VaDE:", compute_purity_average(mean.detach().numpy(), cla, N_CLASS, 2048, 50, method = args.linkage_method))
    #print("PCA:", compute_purity_average(projection, cla, N_CLASS, 2048, 50, method = args.linkage_method))
    #print("Origin:", compute_purity_average(synthetic_data, cla, N_CLASS, 2048, 50, method = args.linkage_method))
    
    print(compute_MW_objective_average(N_CLASS, vae_mean.detach().numpy(), cla, 2048, 50, method = args.linkage_method))
    #print(compute_MW_objective_average(N_CLASS, scaled_mean.detach().numpy(), cla, 2048, 50, method = args.linkage_method))
    print(compute_MW_objective_average(N_CLASS, mean.detach().numpy(), cla, 2048, 50, method = args.linkage_method))
    #print(compute_MW_objective_average(N_CLASS, projection, cla, 2048, 50, method = args.linkage_method))
    #print(compute_MW_objective_average(N_CLASS, synthetic_data, cla, 2048, 50, method = args.linkage_method))
