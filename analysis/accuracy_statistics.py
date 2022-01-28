import os
import argparse
import json
import numpy as np


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

def get_accuracy_statistics(prefix, epoch):
    acc_paths = get_accuracy_paths(prefix, epoch)
    acc = [get_accuracy(x) for x in acc_paths]
    
    mean = np.mean(acc)
    std = np.std(acc)
    
    return mean, std
    

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--prefix', type=str, default='./src/logs/')
    args = parser.parse_args()
    
    mean, std = get_accuracy_statistics(args.prefix, args.epoch)
    print(mean, std)
    
if __name__ == '__main__':
    main()

