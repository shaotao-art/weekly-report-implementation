import transformers
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime


def get_time_str():
    currentDateAndTime = datetime.now()
    day = currentDateAndTime.strftime("%D").replace('/', '-')
    time = currentDateAndTime.strftime("%H:%M:%S")
    currentTime = day + '/' + time
    return currentTime

def get_callbacks(ckp_config):
    checkpoint_callback = ModelCheckpoint(**ckp_config)
    callbacks = []
    callbacks.append(LearningRateMonitor('step'))
    callbacks.append(checkpoint_callback)
    return callbacks

def get_opt_lr_sch(config, model):
    optimizer = optim.AdamW(model.parameters(),
                            **config.optimizer_config)
    if config.lr_sche == 'constant':
        lr_sche = transformers.get_constant_schedule(optimizer)
    if config.lr_sche == 'linear':
        lr_sche = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                **config.lr_sche_config)
    if config.lr_sche == 'cosine':
        lr_sche = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                **config.lr_sche_config)
    if config.lr_sche == 'reduce':
        lr_sche = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        **config.lr_sche_config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_sche,
                'monitor': config.reduce_monitor,
                'interval': 'step'
            }
        }
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': lr_sche,
            'interval': 'step'
        }
    }
     