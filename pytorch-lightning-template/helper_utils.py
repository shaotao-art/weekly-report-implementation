from PIL import Image
from torchvision import transforms as T
from torchvision.utils import make_grid
import numpy as np
import torch
from matplotlib import pyplot as plt



def read_img(img_p):
    img = Image.open(img_p).convert('RGB')
    img = np.array(img)
    return img


def show_or_save_batch_img_tensor(img_tensor, num_sample_per_row, denorm=True, mode='show', save_p=None):
    assert mode in ['show', 'save', 'all']
    if img_tensor.device != torch.device('cpu'):
        img_tensor = img_tensor.cpu()
    if denorm:
        img_tensor = torch.clip((img_tensor + 1.0) / 2.0, 0.0, 1.0)
    img = make_grid(img_tensor, nrow=num_sample_per_row)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    if mode == 'show':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    if mode == 'save':
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    if mode == 'all':
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        assert save_p is not None
        img = Image.fromarray(img)
        img.save(save_p)
        print(f'saving sample img to {save_p}')
    return img
        
    

def read_single_img_for_model_input(img_p):
    img = Image.open(img_p).convert('RGB')
    t = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
            
        ])
    return t(img).unsqueeze(0)
    

def print_model_num_params_and_size(model):
    MB = 1024 * 1024
    cal_num_parameters = lambda module: sum([p.numel() for p in module.parameters() if p.requires_grad == True])
    num_param_to_MB = lambda num_parameters: num_parameters * 4  / MB
    total_num_params = cal_num_parameters(model) 
    print(f'model #params: {total_num_params / (10 ** 6)}M, fp32 model size: {num_param_to_MB(total_num_params)} MB')