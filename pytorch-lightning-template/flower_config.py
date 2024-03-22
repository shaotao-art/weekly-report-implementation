DEBUG = False
# TRAINING
device = 'cuda'

# optimizer and lr_sche
optimizer_config = {
    'lr': 1e-4,
    'weight_decay': 0
}
lr_sche = 'cosine'
warm_up_epoch = 0
num_ep = 20


model_config = dict(
    sample_size=128,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # the number of output channes for each UNet block
    block_out_channels=(64, 128, 256, 256, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    )
)

train_data = 'flower'
train_data_config = dict(
    DEBUG = DEBUG,
    img_size=128,
    normalize_config=dict(
        mean=(0.5, ),
        std=(0.5, )
    ),
    img_root='/root/autodl-tmp/ox-flowers-jpg',
    train_dataloader_config=dict(batch_size=32,
                                 shuffle=True,
                                 num_workers=8)
)


resume_ckpt_path = None # only load model weight from ckp
load_weight_from = None # resume all trainer state

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
# EMA support
use_ema = True
ema_config = dict(
    ema_start_epoch = 0,
    ema_decay = 0.999
)
# LOGGING
enable_wandb = True
wandb_config = dict(
    project='diffusion',
    offline=True if DEBUG == True else False
)

ckp_root = f'[{wandb_config["project"]}]'
