import torch
import pickle

class EMA():
    """WARNING, WE DID NOT TAKE BATCHNORM INTO CONSIDERATION"""
    def __init__(self, model, decay, device='cuda'):
        self.model = model
        self.decay = decay
        self.device = device
        self.ema_weight = {}
        self.origin_weight = {}
        self.init_ema()

    def init_ema(self):
        for name, param in self.model.named_parameters():
            assert 'BatchNorm' not in name
            if param.requires_grad:
                self.ema_weight[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weight
                self.ema_weight[name] = (1.0 - self.decay) * param.data + self.decay * self.ema_weight[name]

    def use_ema_weight(self):
        """copy ema weight to model, and backup model's origin weight"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weight
                self.origin_weight[name] = param.data
                param.data = self.ema_weight[name]
        self.model.to(self.device)

    def restore(self):
        """restore original weight"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.origin_weight
                param.data = self.origin_weight[name]
        self.origin_weight = {}
        self.model.to(self.device)
    
    def save_ema_weight(self, p):
        with open(p, 'wb') as f:
            pickle.dump(self.ema_weight, f)