import os
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms as T

from helper_utils import read_img



def to_tensor_transform(mean, std):
    return T.Compose([
      T.ToTensor(),
      T.Normalize(mean=mean, std=std),
      
        
    ]) 
    
def generative_model_train_transform(img_size):
    return A.Compose([
        A.SmallestMaxSize(img_size),
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5)
    ])
    
    
class FlowerDataset(Dataset):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.img_lst = os.listdir(config.img_root)
    if config.DEBUG:
      self.img_lst = self.img_lst[:100]
    self.img_train_t = generative_model_train_transform(config.img_size)
    self.to_tensor = to_tensor_transform(**config.normalize_config)
    
    
  def __len__(self):
    return len(self.img_lst)
  
  def prepare_img(self, idx):
    img_p = os.path.join(self.config.img_root, self.img_lst[idx])
    img = read_img(img_p)
    transformed_img = self.img_train_t(image=img)['image']
    img_tensor = self.to_tensor(transformed_img)
    return img_tensor
  
  def __getitem__(self, idx):
    img = self.prepare_img(idx)
    return dict(img=img)
    
    
