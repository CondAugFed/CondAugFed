import os
from collections import Counter
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

def load_data(path):
    return torch.load(path)

def load_label(path):
    return load_data(path)[1]

def get_sorted_label(path):
    return sorted(load_label(path))

def create_hist_count(path):
    sorted_label = get_sorted_label(path)
    count = Counter(sorted_label)
    return count

def create_sorted_hist_count(path, num_classes=10):
    sorted_count = []
    count = create_hist_count(path)
    for i in range(num_classes):
        if count.get(i) is not None:
            sorted_count.append(count[i])
        else:
            sorted_count.append(0)
            
    return sorted_count
    
def create_pivot(root, num_classes, num_clients):
    dic = {}
    for i in range(num_clients):
        path = os.path.join(root, f'subset_{i}.pt')
        sorted_count = create_sorted_hist_count(path, num_classes)
        dic[f'Client {i+1}'] = sorted_count
    
    return pd.DataFrame(dic).T

def draw_heatmap(df, save_path):
    sns.heatmap(df, annot=True, fmt='d')
    plt.xlabel('Class ID')
    plt.savefig(save_path)
    
    
mode = 'dirichlet'
data = 'cifar10' # one of MNIST / fMNIST / SVHN / cifar10
root = f'./split_dataset/{mode}/{data}/Train/'

os.makedirs(f'results/data_count', exist_ok=True)
draw_heatmap(create_pivot(root, num_classes=10, num_clients=10), f'results/data_count/{mode}_{data}_count.png')
