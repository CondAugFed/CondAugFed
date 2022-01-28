import os
import argparse
import json
import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

def read_json(path):
    with open(path, 'r') as f:
        result = json.load(f)
    return result    

def get_accuracy(path):
    result = read_json(path)
    try:
        return result['accuracy']
    except:
        raise KeyError

def get_accuracies(paths):
    assert isinstance(paths, list), 'paths should be a list.'
    
    return [get_accuracy(x) for x in paths]
    
def get_accuracy_dirs(prefix, epoch):
    version_dirs = os.listdir(prefix)
    full_dir = lambda x: os.path.join(prefix, x, 'accuracy')
    accuracy_dirs = [full_dir(x) for x in version_dirs]
    return accuracy_dirs

def get_accuracy_paths(prefix, epoch):
    acc_dirs = get_accuracy_dirs(prefix, epoch)
    full_path = lambda x: os.path.join(x, f'classify_result_{epoch}.json')
    accuracy_paths = [full_path(x) for x in acc_dirs]
    return accuracy_paths

def get_accuracy_lists(prefix, epoch):
    acc_paths = get_accuracy_paths(prefix, epoch)
    return [get_accuracy(x) for x in acc_paths]

def get_accuracy_statistics(prefix, epoch):
    acc_paths = get_accuracy_paths(prefix, epoch)
    acc = [get_accuracy(x) for x in acc_paths]
    
    mean = np.mean(acc)
    std = np.std(acc)
    
    return mean, std

def create_dataframe(m_list, s_list, a_list):
    data = {
        'Method': m_list,
        'Sampling ratio': s_list,
        'Accuracy': a_list
    }
    
    df = DataFrame(data)
    return df
    
def stack_list(mode, data, ratio, method):
    
    m_list = []
    s_list = []
    a_list = []
    
    for m in method:
        if m == 'FedAvg':
            for r in ratio:
                if r != 1.0:
                    prefix = f'src/logs/fedavg_ratio/{mode}/{data}/{r}/client_0/'
                else:
                    prefix = f'src/logs/fedavg/{data}/{mode}/client_0/'
                
                a_ = get_accuracy_lists(prefix, epoch=150)
                
                a_list += a_
                m_list += [m] * len(a_)
                s_list += [r] * len(a_)
        else:
            for r in ratio:
                if r != 1.0:
                    prefix = f'src/logs/ca_fedavg_ratio/{mode}/{data}/{r}/client_0/'
                else:
                    prefix = f'src/logs/ca_fedavg/{mode}/{data}/client_0/'
                    
                a_ = get_accuracy_lists(prefix, epoch=150)
                
                a_list += a_
                m_list += [m] * len(a_)
                s_list += [r] * len(a_)
    
    return m_list, s_list, a_list

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=150)
    args = parser.parse_args()
    
    mode = 'dirichlet'
    data = 'SVHN'
    ratio = [0.2, 0.5, 1.0]
    method = ['FedAvg', 'Ours']
    
    m_list, s_list, a_list = stack_list(mode, data, ratio, method)
    df = create_dataframe(m_list, s_list, a_list)

    sns.set_theme(style='darkgrid')
    sns.set_palette('Set2')
    sns.barplot(data=df, x="Sampling ratio", y="Accuracy", hue='Method', ci='sd')
    
    os.makedirs(f'results/sample_ratio_ablation/', exist_ok=True)
    plt.savefig(f'results/sample_ratio_ablation/{mode}_{data}.png')

        
if __name__ == '__main__':
    main()

