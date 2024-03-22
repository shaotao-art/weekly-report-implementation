from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
from mmcv import Config
from pytorch_lightning.loggers import WandbLogger
import os
import argparse
import torch

from utils import get_callbacks, get_time_str, get_opt_lr_sch
from helper_utils import show_or_save_batch_img_tensor
from model import DDPM
from ema import EMA



class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = DDPM(config)
        if config.use_ema:
            assert 'ema_config' in self.config
            self.ema = EMA(self.model, self.config.ema_config.ema_decay)
    
    
    def on_before_zero_grad(self, *args, **kwargs):
        # update ema after optimizer step
        if self.config.use_ema and self.current_epoch >= self.config.ema_config.ema_start_epoch:
            self.ema.update()   
        
    def training_step(self, batch, batch_idx):
        train_loss = self.model.train_loss(batch)
        self.log('train_loss', train_loss)
        return train_loss
    
    def on_train_epoch_start(self) -> None:
        # use ema weight for val
        if self.config.use_ema and self.current_epoch >= self.config.ema_config.ema_start_epoch:
            self.ema.use_ema_weight()   
        if (self.current_epoch + 1) % self.config.trainer_config.val_check_interval == 0:
            imgs = self.model.sample(**self.config.sample_config)
            b_s = imgs.shape[0]
            imgs = show_or_save_batch_img_tensor(imgs, 
                                    int(math.sqrt(b_s)), 
                                    mode='save',
                                    denorm=True,
                                    save_p=os.path.join(self.config.ckp_root, f'ep-{self.current_epoch}.png'),
                                    )
            self.logger.log_image(key=f"ep-{self.current_epoch}", images=[imgs])
        # use original weight for train
        if self.config.use_ema and self.current_epoch >= 5:
            self.ema.restore()   
            self.ema.save_ema_weight(os.path.join(self.config.ckp_root, 'ema.ckp'))

    def configure_optimizers(self):
        return get_opt_lr_sch(self.config, self.model)


def run(args):
    config = Config.fromfile(args.config)
    
    # make ckp accord to time
    time_str = get_time_str()
    config.ckp_root = '-'.join([time_str, config.ckp_root, f'[{args.wandb_run_name}]'])
    config.ckp_config['dirpath'] = config.ckp_root
    config.wandb_run_name = args.wandb_run_name
    # logger
    wandb_logger = None
    if config.enable_wandb:
        wandb_logger = WandbLogger(**config.wandb_config,
                                name=args.wandb_run_name)
        wandb_logger.log_hyperparams(config)
    
    # MODEL
    print('getting model...')
    model = Model(config)
    if 'load_weight_from' in config and config.load_weight_from is not None:
        # only load weights
        state_dict = torch.load(config.load_weight_from, map_location='cpu')['state_dict']
        model = model.load_state_dict(state_dict)
        print(f'loading weight from {config.load_weight_from}')
    print('done.')
    
    #DATA
    print('getting data...')

    if config.train_data == 'flower':
        from flower_dataset import FlowerDataset
        train_data = FlowerDataset(config.train_data_config)
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        **config.train_data_config.train_dataloader_config,
        dataset=train_data)
    print(f'len train_data: {len(train_data)}, len train_loader: {len(train_loader)}.')
    print('done.')

    # lr sche 
    if config.lr_sche in ['linear', 'cosine']:
        config['lr_sche_config'] = dict()
        config.lr_sche_config['num_warmup_steps'] = int(config.warm_up_epoch * \
            len(train_loader))
        config.lr_sche_config['num_training_steps'] = config.num_ep * \
            len(train_loader)
    
    
    callbacks = get_callbacks(config.ckp_config)
    os.makedirs(config.ckp_root, exist_ok=True)
    config.dump(os.path.join(config.ckp_root, 'config.py'))
    
    #TRAINING
    print('staring training...')
    resume_ckpt_path = config.resume_ckpt_path if 'resume_ckpt_path' in config else None
    trainer = pl.Trainer(accelerator=config.device,
                         max_epochs=config.num_ep,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         **config.trainer_config
                         )
    
    trainer.fit(model,
                train_dataloaders=train_loader,
                ckpt_path=resume_ckpt_path
                )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to mmcv config file")
    parser.add_argument("--wandb_run_name", required=True, type=str, help="wandb run name")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(42)
    run(args)