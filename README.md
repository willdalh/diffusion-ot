# Optimal Transport with Diffusion Models

## Datasets

Models used in experiments on high-dimensional data requires the image datasets. The thesis specifies where they are obtained from.

CelebA-HQ-256 must have the following structure: ``src/data/celebahq256/**/<images>``.

AFHQ must have the following structure:``src/data/afhq/**/<images>``.

## Training

To train a model, the following interface is used. Run from the root directory of the repository.

```bash
usage: run_training.py [-h] [--log_name LOG_NAME] [--dataset DATASET] [--epochs EPOCHS] [--save_interval SAVE_INTERVAL] [--batch_size BATCH_SIZE] [--lr LR] [--ema_decay EMA_DECAY] [--n_T N_T] [--beta1 BETA1] [--beta2 BETA2] [--scheduler {linear,cosine,scaled_linear}] [--model_type {unet,single_dim_net}] [--unet_start_channels UNET_START_CHANNELS] [--unet_down_factors UNET_DOWN_FACTORS [UNET_DOWN_FACTORS ...]] [--unet_bot_factors UNET_BOT_FACTORS [UNET_BOT_FACTORS ...]] [--unet_use_attention UNET_USE_ATTENTION] [--pretrained_dirname PRETRAINED_DIRNAME] [--load_only_models LOAD_ONLY_MODELS] [--debug DEBUG] [--debug_slice DEBUG_SLICE] [--mus MUS [MUS ...]] [--sigmas SIGMAS [SIGMAS ...]]

options:
  -h, --help            show this help message and exit
  --log_name LOG_NAME   The directory to log in
  --dataset DATASET     The dataset to use
  --epochs EPOCHS       The number of epochs to train for
  --save_interval SAVE_INTERVAL
                        The number of epochs between saving models
  --batch_size BATCH_SIZE
                        The batch size
  --lr LR               The learning rate
  --ema_decay EMA_DECAY
                        The decay for the exponential moving average
  --n_T N_T             The number of diffusion steps
  --beta1 BETA1         Beta1 for diffusion
  --beta2 BETA2         Beta2 for diffusion
  --scheduler {linear,cosine,scaled_linear}
                        The scheduler to use
  --model_type {unet,single_dim_net}
                        The model architecture to use
  --unet_start_channels UNET_START_CHANNELS
                        The number of channels in the first layer of the UNet
  --unet_down_factors UNET_DOWN_FACTORS [UNET_DOWN_FACTORS ...]
                        The multiplication of channels when downsampling in the UNet
  --unet_bot_factors UNET_BOT_FACTORS [UNET_BOT_FACTORS ...]
                        The multiplication of channels during the bottleneck layers in the UNet
  --unet_use_attention UNET_USE_ATTENTION
                        Not supported: Whether to use attention in the UNet
  --pretrained_dirname PRETRAINED_DIRNAME
                        The name of the directory to load pretrained models from
  --load_only_models LOAD_ONLY_MODELS
                        Name of directory for model states to load, and ignore other arguments from the pretrained session
  --debug DEBUG         Whether to run in debug mode. Activates tqdm
  --debug_slice DEBUG_SLICE
                        The slice to use on the dataset for debugging
  --mus MUS [MUS ...]   The means of the Gaussians. Write '--mus 'X'' for a univar single. Write '--mus X Y' for a univar double. Write '--mus 'X1 Y1' 'X2 Y2'' for a bivar double
  --sigmas SIGMAS [SIGMAS ...]
                        The stds/covariance of the Gaussians. Write '--sigmas X' for univar single. Write '--sigmas X Y' for univar double Write. For bivariate single, write for example '--sigmas 1,0:0,1'. For bivariate double, write for example '--sigmas 1,0:0,1 1,0:0,1'
```

### Commands to train the models used in the experiments

The log directories are provided to show all hyperparameters. The following commands can be used to train each of them. Specify the ``--log_name`` argument to set a fitting name (defaults to ``train_test``). The log names of the provided models are protected and cannot be overwritten.

Models trained on low-dimensional data is first presented, followed by models trained on high-dimensional data.

#### Low1DMix

```bash
python src/run_training.py --dataset gaussian_mixture --batch_size 1024 --lr 0.0003 --beta1 0.000025 --beta2 0.005 --scheduler scaled_linear --model_type single_dim_net --mus -1.25 -0.25 1.5 --sigmas 0.25 0.333 0.1666666
```

#### Low2DSymMix

```bash
python src/run_training.py --dataset gaussian_mixture --batch_size 1024 --lr 0.0003 --beta1 0.0001 --beta2 0.02 --scheduler scaled_linear --model_type single_dim_net --mus '-5 5' '-5 -5' '5 -5' '5 5' --sigmas 0.1,0:0,0.1 0.1,0:0,0.1 0.1,0:0,0.1 0.1,0:0,0.1
```

#### Low2DASymMix

```bash
python src/run_training.py --dataset gaussian_mixture --batch_size 1024 --lr 0.001 --ema_decay 0.992 --beta1 0.0001 --beta2 0.02 --scheduler scaled_linear --model_type single_dim_net --mus '-5 5' '-5 -5' '5 -5' '3 3' --sigmas 0.1,0:0,0.1 0.1,0:0,0.1 0.1,0:0,0.1 0.1,0:0,0.1
```

#### Low2DUnimodal

```bash
python src/run_training.py --dataset gaussian_mixture --batch_size 1024 --lr 0.003 --ema_decay 0.99 --beta1 0.0001 --beta2 0.02 --scheduler scaled_linear --model_type single_dim_net --mus '10 0' --sigmas 1,0:0,1
```

#### Low2DBimodal

```bash
python src/run_training.py --dataset gaussian_mixture --batch_size 1024 --lr 0.001 --ema_decay 0.992 --beta1 0.0001 --beta2 0.02 --scheduler scaled_linear --model_type single_dim_net --mus '-7.5 0' '7.5 0' --sigmas 1,0:0,1 1,0:0,1
```

#### Low2DSCurve

```bash
python src/run_training.py --dataset s_curve_2d_transformed --batch_size 1024 --lr 0.0003 --beta1 0.0001 --beta2 0.02 --scheduler linear --model_type single_dim_net
```

#### Celeb256

```bash
python src/run_training.py --dataset celebahq256 --batch_size 64 --lr 0.00002 --n_T 4000 --beta1 0.00085 --beta2 0.012 --scheduler scaled_linear --model_type unet --unet_start_channels 32 --unet_down_factors 2 4 8 16 32 64 --unet_bot_factors 64 64 32
```

#### Celeb64

```bash
python src/run_training.py --dataset celeba64cropped --batch_size 128 --lr 0.00008 --n_T 1000 --beta1 0.0001 --beta2 0.02 --scheduler linear --model_type unet --unet_start_channels 64 --unet_down_factors 2 4 4 --unet_bot_factors 8 8 4
```

#### AFHQ256

```bash
python src/run_training.py --dataset afhq256 --batch_size 32 --lr 0.0002 --n_T 4000 --beta1 0.00085 --beta2 0.012 --scheduler scaled_linear --model_type unet --unet_start_channels 32 --unet_down_factors 2 4 8 16 32 64 --unet_bot_factors 64 64 32
```

#### AFHQ256Exp1

```bash
python src/run_training.py --dataset afhq256 --batch_size 16 --lr 0.00008 --n_T 4000 --beta1 0.0001 --beta2 0.02 --scheduler linear --model_type unet --unet_start_channels 64 --unet_down_factors 2 4 4 --unet_bot_factors 8 8 4
```

#### AFHQ256Exp2

```bash
python src/run_training.py --dataset afhq256 --batch_size 32 --lr 0.00008 --n_T 4000 --beta1 0.0001 --beta2 0.02 --scheduler linear --model_type unet --unet_start_channels 64 --unet_down_factors 2 4 8 8 --unet_bot_factors 16 16 8
```
