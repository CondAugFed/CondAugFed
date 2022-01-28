from tbparser.summary_reader import SummaryReader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

data = 'CIFAR10'
mode = 'dirichlet'
fl = 'fedavg'

prefixes = [
    f'.src/logs/fedavg/{data}/{mode}/client_0/',
    f'.src/logs/ca_fedavg/{data}/{mode}/client_0/',
]

legends = [
    'FedAvg',
    'FedAvg+CA+'+r'$L_{reg}$'
]

def prepare_reader(log_dir, tag):
    reader = SummaryReader(log_dir, tag_filter=[tag])
    return reader

def avg_read_log(log_dir, tag):
    versions = os.listdir(log_dir)
    dirs = [os.path.join(log_dir, v) for v in versions]
    
    records = [read_log(d, tag) for d in dirs]
    avg_record = sum(records)/len(versions)
    return avg_record

def cat_read_log(log_dir, tag):
    versions = os.listdir(log_dir)
    dirs = [os.path.join(log_dir, v) for v in versions]
    
    records = [read_log(d, tag) for d in dirs]
    return np.concatenate(records, axis=0)

def read_log(log_dir, tag):    
    temp_step = []
    temp_value = []
    reader = prepare_reader(log_dir, tag)
    
    for item in reader:
        temp_step.append(item.step)
        temp_value.append(item.value)
        
        if item.step == 150:
            break
    
    temp_step = np.expand_dims(np.array(temp_step), axis=1)
    temp_value = np.expand_dims(np.array(temp_value), axis=1)
    
    return np.concatenate([temp_step, temp_value], axis=1)

def read_records(log_dirs, tag='Accuracy/Eval'):
    temp_record = []
    for d in log_dirs:
        temp_record.append(cat_read_log(d, tag))
    return temp_record

def draw_lineplot(records, legends):
    sns.set_theme(style='darkgrid')
    sns.set_palette('Set2')

    for record in records:
        df = pd.DataFrame(record, columns=['step', 'value'])
        sns.lineplot(x='step', y='value', data=df, ci='sd')
        
    plt.legend(legends)
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.savefig(f'results/{fl}/{mode}_{data}.png', bbox_inches='tight')
    plt.close()


records = read_records(prefixes)
draw_lineplot(records, legends)
