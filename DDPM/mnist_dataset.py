import torchvision
from torchvision import transforms as tt
from torch.utils.data import Dataset


class MnistDataset(Dataset):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.mnist_transform = tt.Compose([
                            tt.Resize((config.img_size, config.img_size)),
                            tt.ToTensor(),
                            tt.Normalize(**config.normalize_config)
                        ])
    self.dataset = torchvision.datasets.MNIST('.', download=True, transform=self.mnist_transform)


  def __len__(self):
    if self.config.DEBUG:
      return 100
    else:
      return len(self.dataset)
  
  
  def __getitem__(self, idx):
    img = self.dataset[idx][0]
    return dict(img=img)
    
    