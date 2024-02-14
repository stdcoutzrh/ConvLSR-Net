from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.isaid_dataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 60
ignore_index = 255
train_batch_size = 2
val_batch_size = 2
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
accumulate_n = 4
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "convlsrnet-head88-down2"
weights_path = "/home/zrh/datasets/semantic/isaid/log_isaid/{}".format(weights_name)
test_weights_name = weights_name
log_name = "/home/zrh/datasets/semantic/isaid/log_isaid/{}".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
strategy = None

pretrained_ckpt_path = None
resume_ckpt_path = None

#  define the network
from geoseg.model.convlsrnet import Model
net = Model(n_class=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

# define the dataloader

train_dataset = ISAIDDataset(data_root="/home/zrh/datasets/semantic/isaid/train", mode='train',
                                 mosaic_ratio=0.4, transform=train_aug)

val_dataset = ISAIDDataset(data_root="/home/zrh/datasets/semantic/isaid/val",transform=val_aug)
test_dataset = ISAIDDataset(data_root="/home/zrh/datasets/semantic/isaid/val",
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
