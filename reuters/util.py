import numpy as np
import scipy
from scipy import cluster

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm

# evaluation:
def transformation(model, data, rate = 2):
    mean, _ = model.encoder(torch.from_numpy(data).float())
    pred = model.predict(torch.from_numpy(data).float())
    cluster_means = model.mu_c[pred]
    scaled_cluster_means = cluster_means * rate
    scaled_mean = (mean - cluster_means) + scaled_cluster_means
    return scaled_mean.detach()

def compute_purity_average(data, cla, n_class = 8,  num = 1024, repeat = 50, method = "ward",  weighted = True,VERBOSE = False):
    purity = []
    for i in range(repeat):
        if i % 10 == 0 and VERBOSE:
            print("{:4.2f}% finished".format(i/repeat * 100))
        index = np.random.choice(np.arange(len(data)), num)
        Z = linkage(data[index], method)
        if weighted:
            purity.append(compute_purity(Z, cla[index], n_class, [0.4,0.25,0.25,0.1]))
        else:
            purity.append(compute_purity(Z, cla[index], n_class))
    purity = np.array(purity)
    return np.mean(purity), np.std(purity)

def compute_MW_objective_average(n_class, data, cla, num = 1024, repeat = 50, method = "ward", VERBOSE = False):
    MW = []
    for i in range(repeat):
        if i % 10 == 0 and VERBOSE:
            print("{:4.2f}% finished".format(i/repeat * 100))
        index = np.random.choice(np.arange(len(data)), num)

        Z = linkage(cla[index].reshape(-1,1), method)
        rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
        max = compute_objective_gt(num, rootnode, cla[index])
        Z = linkage(data[index], method)
        rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
        MW.append(compute_objective_gt(num, rootnode, cla[index]) / max)
    MW = np.array(MW)
    return np.mean(MW), np.std(MW)

def compute_class_dist(xi, xj):
    dist = 0
    for item_x in xi:
        for item_y in xj:
            dist += np.linalg.norm(item_x - item_y)
    return dist / (len(xi) * len(xj))

def compute_pairwise_dist(data, cla, k, num):
    distance_dict = []
    for i in range(k):
        index1 = np.where(cla == i)
        dist_k = []
        for j in range(k):
            index2 = np.where(cla == j)
            dist_k.append(compute_class_dist(data[index1],data[index2]))
        distance_dict.append(dist_k)
    return np.array(distance_dict)

# VAE training:


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_vae(model, train_loader, epoch = 50, lr = 2e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(epoch):
        train_loss = 0
        for data in train_loader:
            data = data
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          i, train_loss / len(train_loader)))


# VaDE training∩╝Ü
def train(model, train_loader, epoch = 50, lr = 2e-4):
    opti=torch.optim.Adam(model.parameters(),lr = lr)
    epoch_bar=tqdm(range(epoch))
    for epoch in epoch_bar:

        L=0
        for x in train_loader:

            loss=model.ELBO_Loss(x)

            opti.zero_grad()
            loss.backward()
            opti.step()

            L+=loss.detach().numpy()
        print(L/len(train_loader))


# synthetic datset generation

def HGMM(n_class, dim, margin, shift = False):
    margin = margin
    mean = np.zeros((n_class , dim))
    #mean[:(n_class // 2), 0] = margin
    #mean[(n_class // 2):, 0] = -margin
    ratio = n_class // 2
    index = 0
    while ratio != 0:
        for i in range(int(n_class // ratio)):
            mean[i*ratio:(i+1)*ratio, index] = (-1) ** i * margin / (2**index)
        #for i in range(8):
            #mean[i*1:(i+1)*1, 2] = (-1) ** i * margin / 4
        ratio = ratio // 2
        index += 1
    if shift:
        mean[:n_class//2,index:index + index] = mean[:n_class//2,:index]
        mean[:n_class//2,:index] = 0
    return mean

def gen_synthetic(dim, margin, n_class, var, num =100, shift = False):
    mean = HGMM(n_class, dim, margin, shift)
    data = np.random.multivariate_normal(mean[0], np.identity(dim), num)
    cla = np.zeros(num)
    for i in range(1, n_class):
        cla = np.concatenate([cla, i*np.ones(num)])
        data = np.concatenate([data, np.random.multivariate_normal(mean[i], var * np.identity(dim), num)])
    print(data.shape)
    return data, cla

def synthetic_tSNE(n_class, margin, var, dim = 100, num = 100, random_proj = False):
    data, cla = gen_synthetic(dim, margin, n_class, var, num = num)
    if random_proj:
        proj = np.random.randn(dim, dim)
        data = data.dot(proj)
    z = TSNE(n_components=2).fit_transform(data)
    if n_class < 10:
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.scatterplot(z[:, 0], z[:, 1], hue = np.array(cla),  palette = sns.color_palette("Paired", n_class),  legend = "full")
    else:
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.scatterplot(z[:, 0], z[:, 1], hue = np.array(cla), legend = "full")
        
def create_data_loader(size = 400, n_class = 16, margin = 8, var = 1, dim = 100, num_per_class = 2000):
    num_batch = n_class * num_per_class // size
    synthetic_data, cla = gen_synthetic(dim, margin, n_class, var, num_per_class, True)
    train_loader = []
    perm = np.random.permutation(n_class * num_per_class)
    synthetic_data = synthetic_data[perm]
    cla = cla[perm]
    for i in range(size):    
        train_loader.append(torch.from_numpy(synthetic_data[i*num_batch:(i+1)*num_batch]).float())
    return train_loader, synthetic_data, cla

# MW objective related:

class node(cluster.hierarchy.ClusterNode):
    def __init__(self, id, left=None, right=None, dist=0, count=1):
        #super(node, self).__init__(id, left=None, right=None, dist=0, count=1)
        self.id = id
        self.left=left
        self.right=right
        self.dist=dist
        self.count=count
        self.parent = None
        
def create_tree(root):
    if root is None:
        return None
    new_left = create_tree(root.left)
    new_right = create_tree(root.right)
    new_root = node(root.id, new_left, new_right, root.dist, root.count)
    return new_root

def create_par(root, par):
    if root is None:
        return
    root.parent = par
    create_par(root.right, root)
    create_par(root.left, root)
    
    
def DFS(node,res):
    if (node.count == 1):
        res.append(node)
        return
    DFS(node.left,res)
    DFS(node.right,res)


def LCA(node1, node2):
    parent_list = []
    par1 = node1.parent
    while par1 is not None:
        parent_list.append(par1)
        par1 = par1.parent
    par2 = node2.parent
    while par2 not in parent_list:
        par2 = par2.parent
    return par2

def purity(root, cla, target):
    nodes_list = []
    DFS(root, nodes_list)
    target_node = []
    for node in nodes_list:
        if target[node.id] == cla:
            target_node.append(node)
    if len(target_node) == 0:
        return 1
    p = 0
    for i in range(len(target_node)):
        for j in range(i, len(target_node)):
            count = 0
            node1 = target_node[i]
            node2 = target_node[j]
            lca = LCA(node1, node2)
            subtree = []
            DFS(lca, subtree)
            for node in subtree:
                if target[node.id] == cla:
                    count += 1
            p+=(count / len(subtree))
    p /= (len(target_node) * (len(target_node) + 1)) / 2
    return p

def compute_purity(Z, target, target_num = 10, weight = None):
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    root = create_tree(rootnode)
    create_par(root, None)
    p = 0
    for i in range(target_num):
        if weight is not None:
            p += weight[i] * purity(root, i, target)
        else:
            p += 1 / target_num * purity(root, i, target)
    return p


# ------ random cut related ------
    
def list_leaves(node):
    if node.is_leaf():
        return [node.id]
    else:
        result = []
        result += list_leaves(node.left)
        result += list_leaves(node.right)
        return result
    
    
def Gaussian_similarity(x1, x2):
    return np.exp(-1 / 2 * (x1 - x2)**2)

def compute_objective(root, max_obj):
    obj = 0
    if isinstance(root, Leaf_node):
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, xi in enumerate(right_leaves):
            for j in range(i + 1, len(right_leaves)):
                xj = right_leaves[j]
                obj += len(left_leaves) * Gaussian_similarity(xi, xj)
                
        for i, xi in enumerate(left_leaves):
            for j in range(i + 1, len(left_leaves)):
                xj = left_leaves[j]
                obj += len(right_leaves) * Gaussian_similarity(xi, xj)
                
                
        obj_right = compute_objective(root.right, max_obj)
        obj_left = compute_objective(root.left, max_obj)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left
    
def compute_objective_plus(n, root):
    obj = 0
    if isinstance(root, Leaf_node):
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, xi in enumerate(right_leaves):
            for j, xj in enumerate(left_leaves):
                obj += (n - len(left_leaves) - len(right_leaves)) * Gaussian_similarity(xi, xj)
                
        obj_right = compute_objective_plus(n, root.right)
        obj_left = compute_objective_plus(n, root.left)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left 
    
def compute_objective_gt(n, root, cla):
    obj = 0
    if root.is_leaf():
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, index1 in enumerate(right_leaves):
            for j, index2 in enumerate(left_leaves):
                xi = cla[index1]
                xj = cla[index2]
                #if xi == xj:
                obj += (n - len(left_leaves) - len(right_leaves)) * (xi == xj)
                
        obj_right = compute_objective_gt(n, root.right, cla)
        obj_left = compute_objective_gt(n, root.left, cla)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left     

    
def compute_objective_increment(root, whole_data, dic):
    if isinstance(root, Leaf_node):
        return 0
    revenue = 0
    right_leaves  = list_leaves(root.right)
    left_leaves = list_leaves(root.left)
    revenue += (len(left_leaves) + 1) * compute_revenue(right_leaves, whole_data, dic)
    revenue += (len(right_leaves) + 1) * compute_revenue(left_leaves, whole_data, dic)
    rev_right = compute_objective_increment(root.right, whole_data, dic)
    rev_left = compute_objective_increment(root.left, whole_data, dic)
    return revenue + rev_right + rev_left

def max_objective(data):
    data = np.sort(data)
    result = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                result += max(Gaussian_similarity(data[i], data[j]), Gaussian_similarity(data[j], data[k]))
                
    return result 

def min_objective(data):
    n = len(data)
    data = np.sort(data)
    result = 0
    for i in range(n - 1):
        summant = 0
        for j in range(i + 1):
            summant += Gaussian_similarity(data[i + 1], data[j]) 
        result += (n - i - 2) * summant
    return result

def sum_objective(data):
    data = np.sort(data)
    result = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                result += Gaussian_similarity(data[i], data[j]) + Gaussian_similarity(data[j], data[k])
                
    return result 
