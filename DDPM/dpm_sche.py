from tqdm import tqdm
import torch


class DPMSche:
    def __init__(self, 
                 beta_start=1e-4, 
                 beta_end=0.02, 
                 num_train_steps=1000, 
                 num_infer_steps=100, 
                 device='cpu') -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_train_steps = num_train_steps
        self.num_infer_steps = num_infer_steps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # for ddpm
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # for ddim
        self.one_over_sqrt_alpha = torch.sqrt(self.alphas) ** -1
        self.w_on_pred_noise = (1 - self.alphas) / torch.sqrt(1 - self.alphas_cumprod)
        self.w_readd_noise = torch.sqrt(self.betas)

    
    def add_noise(self, img, noise, t):
        assert len(img.shape) == 4
        assert len(noise.shape) == 4
        assert len(t.shape) == 1
        assert t.shape[0] == img.shape[0]
        img = img.to(self.device)
        noise = noise.to(self.device)
        time_steps = t[:, None, None, None]
        noisy_img = torch.sqrt(self.alphas_cumprod[time_steps]) * img + (1 - self.alphas_cumprod[time_steps]) * noise
        return noisy_img
    
    def ddim_one_step_back(self, sample, pred_noise, cur_t):
        prev_t = max(cur_t - (self.num_train_steps // self.num_infer_steps), 0)

        denoised_x_0 = (sample - self.sqrt_one_minus_alphas_cumprod[cur_t] * pred_noise) / self.sqrt_alphas_cumprod[cur_t]
        # clip x_0 instead of x_t-1
        denoised_x_0 = torch.clip(denoised_x_0, -1, 1)
        # cal x_t-1
        if prev_t == 0:
            sample = denoised_x_0
        else:
            sample = self.sqrt_alphas_cumprod[prev_t] * denoised_x_0 + self.sqrt_one_minus_alphas_cumprod[prev_t] * pred_noise 
        return sample
    
    
    @torch.no_grad()
    def ddim_sample(self, model, noise):
        model.eval()
        model.to(self.device)
        sample = noise.to(self.device)
        for iter_ in tqdm(range(self.num_infer_steps), leave=False):
            t = self.num_train_steps - 1 - iter_ * self.num_train_steps // self.num_infer_steps
            pred_noise = model(sample, t).sample
            sample = self.ddim_one_step_back(sample, pred_noise, t)
        return sample
    
    def ddpm_one_step_back(self, x_t, t, noise_pred):
        device = self.device
        if t != 0:
            readd_noise = torch.randn_like(x_t, device=device)
        else:
            readd_noise = torch.zeros_like(x_t, device=device)
            
        mu = self.one_over_sqrt_alpha[t] * (x_t - self.w_on_pred_noise[t] * noise_pred)
        x_t = mu + self.w_readd_noise[t] * readd_noise
        return torch.clip(x_t, -1, 1)
        
        
    @torch.no_grad()
    def ddpm_sample(self, model, noise):
        assert len(noise.shape) == 4
        device = self.device
        noise = noise.to(device)
        model.eval()
        model.to(device)
        x_t = noise
        # always use num train step when sample using ddpm
        for t in reversed(tqdm(range(self.num_train_steps), leave=False)):
            noise_pred = model(x_t, t).sample
            x_t = self.ddpm_one_step_back(x_t, t, noise_pred)
        return x_t      