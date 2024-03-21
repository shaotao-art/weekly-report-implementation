# Common models in Generative Models

## 已完成的
1. `unet(x, time)`
2. `encoder(x, time_embed=None)`
3. `decoder(x, time_embed=None)`
### 说明
1. unet中time为必要参数，为diffusion model设计。
2. unet encoder的大致操作为：先downsample，然后在该resolution上进行不改变分辨率的操作，encoder第一个block没有downsample操作。
3. unet decoder的大致操作为：先upsample，然后在该resolution上进行不改变分辨率的操作，decoder的第一个block没有upsample操作。
4. `2,3`的目的为：在encode时不希望input一进入就downsample，在decode时不希望conv_out之前接一个upsample。
5. encoder和decoder均设计为可接受condition，但是输入为time_embed，而不是未经embed的time，若要实现conditinal_encoder/decoder，需要自行将encoder/decoder的wrapper，并在该wrapper中实现time_embed_func。



## TODO
1. linear attention
2. attention with multihead for image