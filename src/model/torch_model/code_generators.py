import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class EmbeddingCodeGenerator(nn.Module):
    """."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(EmbeddingCodeGenerator, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        class_num = 10
        self.embedding = nn.Embedding(class_num, class_num)
    
    def forward(self, z, y, require_embed=False):
        """z is noise, y is one-hot label"""
        y_idx = torch.nonzero(y)[:,1]
        conditioning_label = self.embedding(y_idx)
        conditioned = torch.cat([z,conditioning_label], dim=1)
        if require_embed:
            self.fc2(F.relu(self.fc1(conditioned))), conditioning_label
        else:
            return self.fc2(F.relu(self.fc1(conditioned)))

class SimpleCodeGenerator(nn.Module):
    """."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleCodeGenerator, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z, y):
        conditioned = torch.cat([z,y], dim=1)
        return self.fc2(F.relu(self.fc1(conditioned)))

class SimpleLinearCodeGenerator(nn.Module):
    """."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleLinearCodeGenerator, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z, y):
        conditioned = torch.cat([z,y], dim=1)
        return self.fc2(self.fc1(conditioned))

class IdentityCodeGenerator(nn.Module):
    """."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(IdentityCodeGenerator, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z, y):
        conditioned = torch.cat([z,y], dim=1)
        return conditioned


class NTXentLoss(nn.Module):
    """."""
    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def __call__(self, anchor_code, negative_code, positive_sample_size=None):
        # batch_size = anchor_code.shape[0]
        if positive_sample_size is None:
            sample_size = anchor_code.shape[0] # use all
        else:
            sample_size = positive_sample_size

        neg_inf_diag = torch.diag(torch.tensor([-np.inf]*sample_size)).detach().to(anchor_code.device)
        anchor_code = anchor_code[:sample_size, :]

        posi_sim = torch.matmul(anchor_code, anchor_code.T)/self.temperature + neg_inf_diag
        nega_sim = torch.matmul(anchor_code, negative_code.T)/self.temperature
        total_sim = torch.cat([posi_sim, nega_sim], dim=1) # Bs x 2Bs

        # max_val = torch.max(
        #         posi_sim, torch.max(nega_sim, dim=1, keepdim=True)[0]
        #     ).detach()

        # log_softmax is much more stable than take softmax followed by log, sequentially.
        # Thank you, Pytorch.
        normalized_negative_log_sim = -F.log_softmax(total_sim, dim=1)
        normalized_negative_log_sim_posi = normalized_negative_log_sim[:sample_size, :sample_size]
        loss = torch.mean(normalized_negative_log_sim_posi)

        return loss
