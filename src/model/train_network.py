import os
import ntpath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from src.data.utils import img2patch, patch2img
import json

import timeit
    
def to_np(tensor):
    """Convert torch.tensor to numpy.ndarray
    The shape will be changed as [B, C, H, W] -> [B, H, W, C]
    """
    if tensor.device == 'cpu':
        return tensor[0,...].permute(1,2,0).numpy()
    else:
        return tensor[0,...].permute(1,2,0).cpu().numpy()
        
def get_loss(loss_name):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    else:
        raise NameError('get_loss: {} is not defined.'.format(loss_name))

class BaseTrainer:
    def __init__(self, device, train_loader, test_loader, opt:dict):
        self.device = device
        self.prepare_dataset(train_loader, test_loader)        
        self.set_attributes_from_dict(opt)
        self.create_log_directories()
        self.writer = SummaryWriter(log_dir=self.logdir)
        
        self.log_save_period = 100 # save period for state dict (unit: epoch)
        self.loss_record_step = 20 # write period for tensorboard (unit: iteration)
        
        self.opt = opt

    def prepare_dataset(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size= train_loader.batch_size
    
        self.train_num = len(self.train_loader.dataset)
        self.test_num = len(self.test_loader.dataset)
        
    def set_attributes_from_dict(self, dict):
        print("="*20)
        print(f'Hyper-parameters of {self.__class__}.')
        for k, v in dict.items():
            setattr(self, k, v)
            print(f'Set self.{k} to be {v}.')
        print("="*20)
        
    def save_checkpoint(self, epoch):
        try:
            checkpoint = {
                'state_dict':self.model.state_dict(),
                'optimizer':self.optim.state_dict()
            }
            checkpoint_path = os.path.join(self.model_save_dir, f'model_state_dict_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            return True
        except Exception as e:
            print('Error detected when saving model: ', e)
            return False
            
    def create_log_directories(self):
        os.makedirs(self.logdir, exist_ok=True)
        self.model_save_dir = os.path.join(self.logdir, 'checkpoints/')
        self.draw_save_dir = os.path.join(self.logdir, 'samples/')
        self.acc_save_dir = os.path.join(self.logdir, 'accuracy/')
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.draw_save_dir, exist_ok=True)
        os.makedirs(self.acc_save_dir, exist_ok=True)
        
    def configure_optimizer(self):
        self.optim = None
    
    def train_step(self):
        pass
    
    def train(self, num_epochs):
        for e in range(1, num_epochs+1):
            self.train_step()

class ClassifierTrainer(BaseTrainer):
    def __init__(self, device, model, train_loader, test_loader, opt:dict):
        super(ClassifierTrainer, self).__init__(device, train_loader, test_loader, opt)
        
        self.model = model.to(device)
        self.configure_optimizer()
        self.loss = get_loss('ce')
        
    def configure_optimizer(self):
        self.optim = torch.optim.SGD(self.model.parameters(), self.lr)

    def train_step(self, epoch, current_round=None):
        lossAvg = 0
        iter_count = 1
        for in_, target_ in tqdm.tqdm(self.train_loader):
            in_ = in_.to(self.device)
            target_ = target_.to(self.device)
            
            self.optim.zero_grad()
            out_ = self.model(in_)
            loss = self.loss(out_, target_)
            loss.backward()
            self.optim.step()

            # record to tensorboard
            if (iter_count+1) % self.loss_record_step == 0:
                loss_value = loss.data.cpu().detach().numpy()
                step = (epoch-1)*len(self.train_loader)+iter_count
                self.writer.add_scalar('Loss/train', loss_value, step)
                self.writer.flush()
                
            iter_count += 1
            lossAvg += loss.data.cpu().detach().numpy() / len(self.train_loader)
        
        return lossAvg

    def train(self, num_epochs, current_round=None):
        self.model.train()
        
        start_epoch = current_round if current_round is not None else 1
        end_epoch = start_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            
            if epoch % self.log_save_period == 0:
                self.save_checkpoint(epoch)

            print(f'[Epoch {epoch}] starts...')
            lossAvg = self.train_step(epoch)
            print(f"[Epoch {epoch}] Average loss :{lossAvg}")
        
            if current_round is None:
                if epoch % 50 == 0:
                    self.test(epoch)
        
    def test(self, epoch, save=True):
        outList = []
        targetList = []
        self.model.eval()

        for in_, target_ in tqdm.tqdm(self.test_loader):
            in_ = in_.to(self.device)
            target_ = target_.to(self.device)
            with torch.no_grad():
                out_ = self.model(in_)
                out_maxidx = torch.argmax(out_, dim=1)
                outList += list(out_maxidx.detach().cpu().numpy())
                targetList += list(target_.detach().cpu().numpy())
            
            del out_

        # calculate accuracy, sensitivity, ...
        result = classification_report(targetList, outList)
        result_dict = classification_report(targetList, outList, output_dict=True)
        print(result)
        if save:
            json_save_path = os.path.join(self.acc_save_dir, f'classify_result_{epoch}.json')
            with open(json_save_path, "w") as json_file:
                json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        return result_dict

class CAFedAvgTrainer(ClassifierTrainer):
    def __init__(self, device, model, generator, code_generator, train_loader, test_loader, opt):
        super(CAFedAvgTrainer,self).__init__(device, model, train_loader, test_loader, opt)
    
        self.generator = generator.to(device)
        self.code_generator = code_generator.to(device)
        # do not update generator.
        self.generator = self.no_require_grad(self.generator)
        self.code_generator = self.no_require_grad(self.code_generator)
        
        self.load_generator(load_epoch=self.gen_load_epoch)
        
    def no_require_grad(self, model):
        for p in model.parameters():
            p.requires_grad = False
        return model
    
    def generate_random_label(self):
        '''return randomly (uniform) generated labels as one-hot vector'''
        # Target labels
        class_idx = torch.randint(0, self.class_num, (self.batch_size,))
        class_onehot = F.one_hot(class_idx, self.class_num).type(torch.float32)
        class_idx = list(class_idx.numpy())

        return class_onehot, class_idx

    def _sample(self, noise, condition):
        code = self.code_generator(z=noise, y=condition)
        return self.generator(code).detach()

    def is_real_data(self):
        uniform_sample = torch.rand(1).item()
        if uniform_sample > self.real_percentage:
            return False
        else:
            return True

    def load_generator(self, load_epoch=200):
        checkpoint_path = os.path.join(self.gen_load_dir, 'model_state_dict_{}.pt'.format(load_epoch))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['g_state_dict'])
        self.code_generator.load_state_dict(checkpoint['cg_state_dict'])

    def configure_optimizer(self):
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train_step(self, epoch, current_round=None):
        lossAvg = 0
        iter_count = 1
        for in_, target_ in tqdm.tqdm(self.train_loader):
            if self.is_real_data():
                in_ = in_.to(self.device)
                target_ = target_.to(self.device)
            else:
                target_onehot, target_ = self.generate_random_label()
                target_onehot = target_onehot.to(self.device)
                noise = torch.randn((self.batch_size, 128)).to(self.device)
                in_ = self._sample(noise, target_onehot)
                target_ = torch.tensor(target_).to(self.device)
            
            self.optim.zero_grad()
            out_ = self.model(in_)
            loss = self.loss(out_, target_)
            loss.backward()
            self.optim.step()

            # record to tensorboard
            if (iter_count+1) % self.loss_record_step == 0:
                loss_value = loss.data.cpu().detach().numpy()
                step = (epoch-1)*len(self.train_loader)+iter_count
                self.writer.add_scalar('Loss/train', loss_value, step)
                self.writer.flush()
            
            iter_count += 1
            lossAvg += loss.data.cpu().detach().numpy() / len(self.train_loader)
        
        return lossAvg

    def train(self, num_epochs, current_round=None):
        self.model.train()
        
        start_epoch = current_round if current_round is not None else 1
        end_epoch = start_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            
            # In this setting, we need to save the global model of certain round before local update.
            if epoch % self.log_save_period == 0:
                self.save_checkpoint(epoch)

            print(f'[Epoch {epoch}] starts...')
            lossAvg = self.train_step(epoch)
            print(f"[Epoch {epoch}] Average loss :{lossAvg}")
        

class FedProxTrainer(ClassifierTrainer):
    def __init__(self, device, model, train_loader, test_loader, opt):
        super(FedProxTrainer, self).__init__(device, model, train_loader, test_loader, opt)

        self.global_params = []
  
    def train_step(self, epoch, current_round=None):
        lossAvg = 0
        iter_count = 1
        for in_, target_ in tqdm.tqdm(self.train_loader):
            in_ = in_.to(self.device)
            target_ = target_.to(self.device)
            
            self.optim.zero_grad()
            out_ = self.model(in_)
            loss = self.loss(out_, target_)
            
            #for fedprox
            fed_prox_reg = 0.0
            
            for param_index, (k,param) in enumerate(self.model.state_dict().items()):
                if not 'num_batches_tracked' in k:
                    fed_prox_reg += ((self.mu / 2) * torch.norm((param - torch.tensor(self.global_params[param_index]).to(self.device)))**2)
            loss += fed_prox_reg
            
            loss.backward()
            self.optim.step()

            # record to tensorboard
            if (iter_count+1) % self.loss_record_step == 0:
                loss_value = loss.data.cpu().detach().numpy()
                step = (epoch-1)*len(self.train_loader)+iter_count
                self.writer.add_scalar('Loss/train', loss_value, step)
                self.writer.flush()
            
            iter_count += 1
            lossAvg += loss.data.cpu().detach().numpy() / len(self.train_loader)

            del param, fed_prox_reg
        
        return lossAvg

class CAFedProxTrainer(CAFedAvgTrainer):
    def __init__(self, device, model, generator, code_generator, train_loader, test_loader, opt):
        super(CAFedProxTrainer, self).__init__(device, model, generator, code_generator, train_loader, test_loader, opt)
        self.global_params = []
        
    def train_step(self, epoch, current_round=None):
        lossAvg = 0
        iter_count = 1
        for in_, target_ in tqdm.tqdm(self.train_loader):
            if self.is_real_data():
                in_ = in_.to(self.device)
                target_ = target_.to(self.device)
            else:
                target_onehot, target_ = self.generate_random_label()
                target_onehot = target_onehot.to(self.device)
                noise = torch.randn((self.batch_size, 128)).to(self.device)
                in_ = self._sample(noise, target_onehot)
                target_ = torch.tensor(target_).to(self.device)
            
            self.optim.zero_grad()
            out_ = self.model(in_)
            loss = self.loss(out_, target_)
            
            #for fedprox
            fed_prox_reg = 0.0
            
            for param_index, (k,param) in enumerate(self.model.state_dict().items()):
                if not 'num_batches_tracked' in k:
                    fed_prox_reg += ((self.mu / 2) * torch.norm((param - torch.tensor(self.global_params[param_index]).to(self.device)))**2)
            loss += fed_prox_reg
            
            loss.backward()
            self.optim.step()

            # record to tensorboard
            if (iter_count+1) % self.loss_record_step == 0:
                loss_value = loss.data.cpu().detach().numpy()
                step = (epoch-1)*len(self.train_loader)+iter_count
                self.writer.add_scalar('Loss/train', loss_value, step)
                self.writer.flush()
            
            iter_count += 1
            lossAvg += loss.data.cpu().detach().numpy() / len(self.train_loader)

            del param, fed_prox_reg
        
        return lossAvg
