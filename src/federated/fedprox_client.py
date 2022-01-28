import argparse
import os
from omegaconf import OmegaConf
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.torch_model.cnn_backbone import MNISTCNN32
from torchvision import models
from src.model.train_network import FedProxTrainer as Trainer
from src.data.splited_dataset import getMnistDataLoader as mnist_loader
from src.data.splited_dataset import getCifar10DataLoader as cifar_loader
from src.data.splited_dataset import getfMNISTDataLoader as fmnist_loader
from src.data.splited_dataset import getSVHNDataLoader as svhn_loader

class Client(fl.client.NumPyClient):
    def __init__(self, trainer, local_epochs):
        super().__init__()
        self.trainer = trainer
        self.net = self.trainer.model
        self.local_epochs = local_epochs

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)

        state_dict = {}

        for k, v in params_dict:
            if 'num_batches_tracked' in k:
                state_dict[k] = torch.Tensor([v[()]])
            else:
                state_dict[k] = torch.Tensor(v)

        state_dict = OrderedDict(state_dict)

        self.net.load_state_dict(state_dict, strict=True)
        self.trainer.global_params = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.train(num_epochs=self.local_epochs, current_round=config['round'])
        
        if self.trainer.train_num is not None:
            return self.get_parameters(), int(self.trainer.train_num), {}
        else:
            data_num = len(self.trainer.train_loader) * self.trainer.train_loader.batch_size
            return self.get_parameters(), int(data_num), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if config['round'] % 5 == 0:
            result = self.trainer.test(config['round'])
            final_round = config['num_rounds'] == config['round'] 

            if config['round'] >= 0:
                self.trainer.writer.add_scalar('Accuracy/Eval', result['accuracy'], config['round'])
                self.trainer.writer.flush()

            if final_round:
                self.trainer.writer.add_hparams(self.trainer.opt, {'accuracy':result['accuracy']})
                self.trainer.writer.flush()

            if self.trainer.test_num is not None:
                return 0.0, int(self.trainer.test_num), {"accuracy":result['accuracy']}
            else:
                data_num = len(self.trainer.test_loader) * self.trainer.test_loader.batch_size
                return 0.0, int(data_num), {"accuracy":result['accuracy']}
        else:
            data_num = len(self.trainer.test_loader) * self.trainer.test_loader.batch_size
            return 0.0, int(data_num), {"accuracy":0.0}


def isint(s):
    try:
        int_s = int(s)
        return True
    except:
        return False

def parse_version_num(directory):
    try:
        dir_list = os.listdir(directory)
        sorted_dir_list = sorted(dir_list)
        sorted_dir_list = [x for x in sorted_dir_list if isint(x.strip('version_'))]
        last_version = sorted_dir_list[-1].split('_')[-1]
        current_version = int(last_version) + 1
    except:
        current_version = 0
    return str(current_version)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--root', type=str, default='.', 
                        help="data root directory that contains 'split_dataset' folder.")
    parser.add_argument('--data', type=str, default='CIFAR10', 
                        help="MNIST / fMNIST / SVHN / CIFAR10")
    parser.add_argument('--subset', type=int, default=0, 
                        help='one integer between 0 and 9')
    parser.add_argument('--split_mode', type=str, default='dirichlet', 
                        help="extreme / dirichlet")
    parser.add_argument('--logdir', type=str, default='src/logs/test/', 
                        help="directory to save logs/checkpoints.")
    parser.add_argument('--server_address', type=str, default='localhost:8080')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_local_epochs', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--log_save_period', type=int, default=20)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-2)
    
    args = parser.parse_args()
    
    args.logdir = os.path.join(args.logdir, f'client_{args.subset}')
    os.makedirs(args.logdir, exist_ok=True)
    version = parse_version_num(args.logdir)
    args.logdir = os.path.join(args.logdir, 'version_'+version)
    
    DEVICE = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available() and args.gpu != 'None') else "cpu")
    if args.data == "MNIST":
        train_loader = mnist_loader(args.root, args.subset, args.batch_size, 0, [32,32], mode=args.split_mode, train=True)
        test_loader = mnist_loader(args.root, 0, args.batch_size, 0, [32,32], mode=args.split_mode, train=False)
    elif args.data == 'fMNIST':
        train_loader = fmnist_loader(args.root, args.subset, args.batch_size, 0, [32,32], mode=args.split_mode, train=True)
        test_loader = fmnist_loader(args.root, 0, args.batch_size, 0, [32,32], mode=args.split_mode, train=False)
    elif args.data == 'CIFAR10':
        train_loader = cifar_loader(args.root, args.subset, args.batch_size, 0, [32,32], mode=args.split_mode, train=True)
        test_loader = cifar_loader(args.root, 0, args.batch_size, 0, [32,32], mode=args.split_mode, train=False)
    elif args.data == 'SVHN':
        train_loader = svhn_loader(args.root, args.subset, args.batch_size, 0, [32,32], mode=args.split_mode, train=True)
        test_loader = svhn_loader(args.root, 0, args.batch_size, 0, [32,32], mode=args.split_mode, train=False)

    if args.data in ['MNIST', 'fMNIST']:
        model = MNISTCNN32()
    elif args.data in ['CIFAR10', 'SVHN']:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        
    opt = vars(args)
    trainer = Trainer(DEVICE, model, train_loader, test_loader, opt)
    trainer.model.to(DEVICE)
    client = Client(trainer, args.num_local_epochs)
    print('client is made')
    fl.client.start_numpy_client(args.server_address, client=client)

if __name__ == '__main__':
    main()