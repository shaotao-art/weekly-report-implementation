import torch 
from torch import nn
from diffusers import UNet2DModel
from dpm_sche import DPMSche


class DDPM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = UNet2DModel(**config.model_config)
        self.sche = DPMSche(**config.sche_config)
        self.loss_fn = nn.MSELoss()
  
        
    def forward(self, noisy_image, t):
        return self.model(noisy_image, t).sample
    

    
    def train_loss(self, batch):
        imgs= batch['img']
        b_s = imgs.shape[0]
        # add noise
        noise = torch.randn_like(imgs, device=self.config.device)
        t = torch.randint(low=0, high=self.sche.num_train_steps, size=(b_s, ), device=self.config.device)
        noisy_image = self.sche.add_noise(imgs, noise, t)
        
        pred_noise = self.model(noisy_image, t).sample
        train_loss = self.loss_fn(pred_noise, noise)
        return train_loss
    
    @torch.no_grad()
    def sample(self, b_s, sample_sche):
        assert sample_sche in ['ddim', 'ddpm']
        self.eval()
        noise = torch.randn((b_s, self.config.model_config.in_channels, self.config.model_config.sample_size, self.config.model_config.sample_size), device=self.config.device)
        if sample_sche == 'ddim':
            imgs = self.sche.ddim_sample(self.model, noise)
        elif sample_sche == 'ddpm':
            imgs = self.sche.ddpm_sample(self.model, noise)
        return imgs
        