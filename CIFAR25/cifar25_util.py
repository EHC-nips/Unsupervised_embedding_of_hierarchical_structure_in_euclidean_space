import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset

import numpy as np

import scipy
from scipy import cluster

from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
from sklearn.decomposition import PCA
import sklearn


def prepare_cifar100(data_dir='./data/cifar100/'):
    super_class = {"beaver": 0, "dolphin": 1, "otter": 2, "seal": 3, "whale": 4,
    "orchid": 5, "poppy": 6, "rose":7 , "sunflower": 8, "tulip":9,
    "apple":10, "mushroom":11, "orange":12, "pear":13, "sweet_pepper":14,
    "cloud":15, "forest":16, "mountain":17, "plain":18, "sea":19,
    "bicycle":20, "bus":21, "motorcycle":22, "pickup_truck":23, "train":24}

    train=CIFAR100(root=data_dir,train=True,download=True)
    test=CIFAR100(root=data_dir,train=False,download=True)

    train_super_class_targets = []
    test_super_class_targets = []
    train_X = []
    test_X = []

    for i in range(len(train.targets)):
        if train.classes[train.targets[i]] not in super_class.keys():
            continue
        cla = int(super_class[train.classes[train.targets[i]]])
        train_super_class_targets.append(cla)
        train_X.append(train.data[i])

    for i in range(len(test.targets)):
        if test.classes[test.targets[i]] not in super_class.keys():
            continue
        cla = int(super_class[test.classes[test.targets[i]]])
        test_super_class_targets.append(cla)
        test_X.append(test.data[i])  

    train_X = np.array(train_X)
    test_X = np.array(test_X)
    X=torch.cat([torch.from_numpy(train_X).float().view(-1,3,32,32)/255.,torch.from_numpy(test_X).float().view(-1,3,32,32)/255.],0)
    Y=torch.cat([torch.Tensor(train_super_class_targets).int(),torch.Tensor(test_super_class_targets).int()],0)
    return X, Y

def gen_cifar100_subset(X,Y,subsample_num = 64):
    cifar100_data = []
    for i in range(25):
        index = np.where(Y.detach().numpy() == i)
        cifar100_data.append(X[index])
    test_X = torch.Tensor([])
    test_Y = np.array([])
    for i in range(25):
        index = np.random.choice(np.arange(600), subsample_num)
        test_X = torch.cat([test_X, cifar100_data[i][index]])
        test_Y = np.concatenate([test_Y, i * np.ones(subsample_num)])
    test_X.shape, test_Y.shape

    h_Y = np.zeros_like(test_Y)
    h_Y[np.where(test_Y >= 5)] = 1
    h_Y[np.where(test_Y >= 10)] = 2
    h_Y[np.where(test_Y >= 15)] = 3
    h_Y[np.where(test_Y >= 20)] = 4
    
    return test_X, test_Y, h_Y


# evaluation:
def transformation(model, data, rate = 5):
    mean, _ = model.encoder(torch.from_numpy(data).float())
    pred = model.predict(torch.from_numpy(data).float())
    cluster_means = model.mu_c[pred]
    scaled_cluster_means = cluster_means * rate
    scaled_mean = (mean - cluster_means) + scaled_cluster_means
    return scaled_mean.detach()

def compute_purity_average(model, data, cla, n_class = 10, num = 100, repeat = 50, method = "ward", eval = "VaDE", transform = False, VERBOSE = False, super_class = True):
    purity = []
    print("repeat:", repeat)
    cifar100_data = []
    for i in range(25):
        index = np.where(cla.detach().numpy() == i)
        cifar100_data.append(data[index])
    for i in range(repeat):
        test_X = torch.Tensor([])
        test_Y = np.array([])
        for j in range(25):
            index = np.random.choice(np.arange(600), num)
            test_X = torch.cat([test_X, cifar100_data[j][index]])
            test_Y = np.concatenate([test_Y, j * np.ones(num)])

        h_Y = np.zeros_like(test_Y)
        h_Y[np.where(test_Y >= 5)] = 1
        h_Y[np.where(test_Y >= 10)] = 2
        h_Y[np.where(test_Y >= 15)] = 3
        h_Y[np.where(test_Y >= 20)] = 4
        
        data = test_X.detach().numpy()
        
        if super_class:
            cla = h_Y
        else:
            cla = test_Y
        if i % 10 == 0 and VERBOSE:
            print("{:4.2f}% finished".format(i/repeat * 100))
        if eval == "VAE":
            _, eval_data, _ = model(torch.from_numpy(data).float())
            eval_data = eval_data.detach().numpy()
        if eval == "PCA":
            pca = PCA(n_components = 10)
            eval_data = pca.fit_transform(data.reshape(-1,32*32*3))
        if eval == "VaDE":
            if transform:
                eval_data = transformation(model, data)
            else:
                eval_data, _ = model.encoder(torch.from_numpy(data).float())
                eval_data = eval_data.detach().numpy()
        if eval == "Origin":
            eval_data = data.reshape(-1,32*32*3)
        Z = linkage(eval_data, method)
        purity.append(compute_purity(Z, cla, n_class))
    purity = np.array(purity)
    return np.mean(purity), np.std(purity)

def compute_MW_objective_average(model, data, cla, n_class = 10, c_num = 100, repeat = 50, method = "ward", eval = "VaDE", transform = False, VERBOSE = False, super_class = False):
    similarity_matrix = np.zeros((25,25))
    for i in range(5):
        similarity_matrix[i*5:(i+1)*5, i*5:(i+1)*5] = np.ones((5,5)) + np.identity(5)
    MW = []
    Y = cla
    cifar100_data = []
    for i in range(25):
        index = np.where(cla.detach().numpy() == i)
        cifar100_data.append(data[index])
    num = c_num * 25
    print("repeat:", repeat)
    for i in range(repeat):
        test_X = torch.Tensor([])
        test_Y = np.array([])
        for i in range(25):
            index = np.random.choice(np.arange(600), c_num)
            test_X = torch.cat([test_X, cifar100_data[i][index]])
            test_Y = np.concatenate([test_Y, i * np.ones(c_num)])

        h_Y = np.zeros_like(test_Y)
        h_Y[np.where(test_Y >= 5)] = 1
        h_Y[np.where(test_Y >= 10)] = 2
        h_Y[np.where(test_Y >= 15)] = 3
        h_Y[np.where(test_Y >= 20)] = 4
        
        data = test_X.detach().numpy()
        if super_class:
            cla = h_Y
        else:
            cla = test_Y
        
        if i % 10 == 0 and VERBOSE:
            print("{:4.2f}% finished".format(i/repeat * 100))
        if eval == "VAE":
            _, eval_data, _ = model(torch.from_numpy(data).float())
            eval_data = eval_data.detach().numpy()
        if eval == "PCA":
            pca = PCA(n_components = 10)
            eval_data = pca.fit_transform(data.reshape(-1, 3*32*32))
        if eval == "VaDE":
            if transform:
                eval_data = transformation(model, data)
            else:
                eval_data, _ = model.encoder(torch.from_numpy(data).float())
                eval_data = eval_data.detach().numpy()
        if eval == "Origin":
            eval_data = data.reshape(-1, 3*32*32)
        Z = linkage(cla.reshape(-1,1), method)
        rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
        max = compute_objective_gt(num, rootnode, cla)
        Z = linkage(eval_data, method)
        rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
        MW.append(compute_objective_gt(num, rootnode, cla) / max)
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

# VaDE training：
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

def HGMM(n_class, dim, margin):
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
    return mean

def gen_synthetic(dim, margin, n_class, var, num =100):
    mean = HGMM(n_class, dim, margin)
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
    synthetic_data, cla = gen_synthetic(dim, margin, n_class, var, num_per_class)
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

def compute_purity(Z, target, target_num = 10):
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    root = create_tree(rootnode)
    create_par(root, None)
    p = 0
    for i in range(target_num):
        p += purity(root, i, target)
    return p/target_num


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
    
def compute_objective_gt(n, root, cla, similarity_matrix = None):
    obj = 0
    if root.is_leaf():
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, index1 in enumerate(right_leaves):
            for j, index2 in enumerate(left_leaves):
                xi = int(cla[index1])
                xj = int(cla[index2])
                #if xi == xj:
                if similarity_matrix is not None:
                    obj += (n - len(left_leaves) - len(right_leaves)) * similarity_matrix[xi, xj]
                else:
                    obj += (n - len(left_leaves) - len(right_leaves)) * (xi == xj)
                
        obj_right = compute_objective_gt(n, root.right, cla)
        obj_left = compute_objective_gt(n, root.left, cla)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left 

