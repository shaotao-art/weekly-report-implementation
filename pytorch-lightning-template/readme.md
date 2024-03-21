# pytorch lightning template

## Introduction
1. config文件为`flower_config.py`, 使用mmcv的config，主要考虑的点有：`lr_sche`,`checkpoint`, `wandb logging`.
2. `model.py`包含了model的实现，`forward`,`predict`等方法均应该在该文件中实现，`run.py`的model应该只作为一个wrapper来使用，实现logging等功能。
## 使用

```
python run.py --config 'path_to_your_config_file' --wandb_run_name 'exp_run_name'
```
## TODO
1. 该template只考虑了训练集，未包含val。
2. 仅考虑了图像。
