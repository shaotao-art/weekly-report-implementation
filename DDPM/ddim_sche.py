from tqdm import tqdm
import torch


class DDIMSche:
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
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    
    def add_noise(self, img, noise, t):
        assert len(img.shape) == 4
        assert len(noise.shape) == 4
        assert len(t.shape) == 1
        assert t.shape[0] == img.shape[0]
        img = img.to(self.device)
        noise = noise.to(self.device)
        time_steps = t[:, None, None, None]
        noisy_img = torch.sqrt(self.alphas_cumprod[time_steps]) * img + torch.sqrt(1 - self.alphas_cumprod[time_steps]) * noise
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