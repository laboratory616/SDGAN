# Improving the Spatial Resolution of Solar Images Using Super-resolution Diffusion GANs

## Training Super-resolution Diffusion GANs ##
We use the following commands to train Super-resolution Diffusion GANs on the solar image dataset.
```
python3 train_sdgan.py --image_size 256 --exp ddgan_vgg_ T4 --num_channels 3 --num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 500 --ngf 64 \
--embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --lr_d 1e-5 --lr_g 1.6e-5 \
--lazy_reg 10 --num_process_per_node 1 --save_content
```
## Evaluation ##
```
python3 test_sdgan_LR.py --image_size 256  --num_channels 3 --num_channels_dae 64  --ch_mult 1 1 2 2 4 4  \
--num_timesteps 4 --num_res_blocks 2  --net_type vgg_lr_T2 --batch_size 1 --epoch_id $EPOCH
```
