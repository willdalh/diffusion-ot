from collections import namedtuple

TrainingArgsType = namedtuple("TrainingArgsType", ["log_name", "log_dir", "dataset", "epochs", "save_interval" "batch_size", "lr", "n_T", "beta1", "beta2", "scheduler", "unet_use_attention", "data_shape", "pretrained_dirname", "pretrained_model"])
