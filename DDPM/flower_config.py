DEBUG = False
# TRAINING
device = 'cuda'

# optimizer and lr_sche
optimizer_config = {
    'lr': 5e-5,
    'weight_decay': 0
}
lr_sche = 'constant'
# warm_up_epoch = 0.5
num_ep = 20
img_size = 64


model_config = dict(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D',
                      'AttnDownBlock2D'),
    up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D',
                    'UpBlock2D'))

# DDPM 
sche_config = dict(
    beta_start=1e-4, 
    beta_end=0.02, 
    num_train_steps=1000, 
    num_infer_steps=100, 
    device=device
)

train_data = 'flower'
train_data_config = dict(
    DEBUG = DEBUG,
    img_size=img_size,
    normalize_config=dict(
        mean=(0.5, ),
        std=(0.5, )
    ),
    img_root='/root/autodl-tmp/ox-flowers-jpg',
    train_dataloader_config=dict(batch_size=20,
                                 shuffle=True,
                                 num_workers=8)
)


sample_config = dict(
    b_s = 16,
    sample_sche = 'ddim'
)


# resume_ckpt_path = 'DDPM/03-22-24/16:08:08-[gen-models]-[flower-hf-unet2d-model]/last.ckpt'
load_weight_from = '/root/autodl-tmp/DDPM/03-22-24/16:08:08-[gen-models]-[flower-hf-unet2d-model]/last.ckpt'

# ckp
ckp_config = dict(
   save_last=True, 
   every_n_epochs=None
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='16-mixed',
    val_check_interval=1
)
use_ema = False
# ema_config = dict(
#     ema_start_epoch = 0,
#     ema_decay = 0.999
# )
# LOGGING
enable_wandb = True
wandb_config = dict(
    project='gen-models',
    offline=True if DEBUG == True else False
)

ckp_root = f'[{wandb_config["project"]}]'
