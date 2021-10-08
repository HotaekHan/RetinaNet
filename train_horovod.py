# python
import os
import argparse
import random
import numpy as np
import shutil

# pytorch
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch import autograd

# distributed
import torch.utils.data.distributed
import horovod.torch as hvd

# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# user-defined
from loss import FocalLoss
from models.retinanet import load_model
from datagen import jsonDataset, ConcatBalancedDataset
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)
start_epoch = 1  # start from epoch 0 or last epoch
fp16 = False
num_workers = 0

# Horovod: initialize library.
hvd.init()

# make output folder
if hvd.rank() == 0:
    if not os.path.exists(config['model']['exp_path']):
        os.mkdir(config['model']['exp_path'])

    if not os.path.exists(os.path.join(config['model']['exp_path'], 'config.yaml')):
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))
    else:
        os.remove(os.path.join(config['model']['exp_path'], 'config.yaml'))
        shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

# set random seed
random.seed(config['params']['random_seed'])
np.random.seed(config['params']['random_seed'])
torch.manual_seed(config['params']['random_seed'])

# variables
best_valid_loss = float('inf')

global_iter_train = 0
global_iter_valid = 0

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(hvd.local_rank())
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")
if config['cuda']['using_cuda']:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(config['params']['random_seed'])

# Horovod: limit # of CPU threads to be used per worker.
# torch.set_num_threads(num_workers)

# tensorboard
if hvd.rank() == 0:
    summary_writer = SummaryWriter(os.path.join(config['model']['exp_path'], 'log'))

# Data
target_classes = utils.read_txt(config['params']['classes'])
num_classes = len(target_classes)
img_size = config['params']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

# Data
print('==> Preparing data..')
bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
train_transforms = A.Compose([
    A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    A.HorizontalFlip(p=0.5),
    # A.OneOf([
    #     A.Sequential([
    #         A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    #     ], p=1.0),
    #     A.Sequential([
    #         A.RandomSizedBBoxSafeCrop(height=img_size[0], width=img_size[1], p=1.0),
    #     ], p=1.0)
    # ], p=1.0),

    A.OneOf([
        A.Sequential([
            A.GaussNoise(var_limit=(100, 150), p=0.5),
            A.MotionBlur(blur_limit=17, p=0.5)
        ], p=1.0),
        A.Sequential([
            A.GaussNoise(var_limit=(100, 150), p=0.5),
            A.MotionBlur(blur_limit=17, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
        ], p=1.0),
        A.Sequential([
            A.GaussNoise(var_limit=(100, 150), p=0.5),
            A.MotionBlur(blur_limit=17, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
            A.ChannelShuffle(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.15, 0.15), rotate_limit=30, p=0.5,
            #                    border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=1.0)
    ], p=1.0),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=bbox_params, p=1.0)
valid_transforms = A.Compose([
    A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], bbox_params=bbox_params, p=1.0)

train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes,
                            transform=train_transforms,
                            input_image_size=img_size,
                            num_crops=config['params']['num_crops'])

valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes,
                            transform=valid_transforms,
                            input_image_size=img_size,
                            num_crops=config['params']['num_crops'])

assert train_dataset
assert valid_dataset

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['params']['batch_size'],
    num_workers=num_workers,
    collate_fn=train_dataset.collate_fn,
    pin_memory=True,
    sampler=train_sampler)

# Horovod: use DistributedSampler to partition the valid data.
valid_sampler = torch.utils.data.distributed.DistributedSampler(
    valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config['params']['batch_size'],
    num_workers=num_workers,
    collate_fn=valid_dataset.collate_fn,
    pin_memory=True,
    sampler=valid_sampler)

# Model
num_classes = len(target_classes)
num_anchors = train_dataset.data_encoder.num_anchors

net = load_model(num_classes=num_classes,
                 num_anchors=num_anchors,
                 basenet=config['params']['base'],
                 is_pretrained_base=config['params']['pre_base'])
net = net.to(device)

# print out net
if hvd.rank() == 0:
    print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

# loss
criterion = FocalLoss(num_classes=num_classes)

# optimizer
if config['params']['optimizer'] == 'SGD':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=float(config['params']['lr']),
                          momentum=0.9, weight_decay=5e-4)
elif config['params']['optimizer'] == 'Adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=float(config['params']['lr']))
else:
    raise ValueError('not supported optimizer')

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if fp16 else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum)

# set lr scheduler
if config['params']['lr_multistep'] != 'None':
    milestones = config['params']['lr_multistep']
    milestones = milestones.split(', ')
    for iter_milestone in range(len(milestones)):
        milestones[iter_milestone] = int(milestones[iter_milestone])
    scheduler_for_lr = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
else:
    scheduler_for_lr = None

# set pre-trained
if config['model']['model_path'] != 'None' and hvd.rank() == 0:
    print('loading pretrained model from %s' % config['model']['model_path'])
    ckpt = torch.load(config['model']['model_path'], map_location=device)
    weights = utils._load_weights(ckpt['net'])
    missing_keys = net.load_state_dict(weights, strict=False)
    print(missing_keys)
    start_epoch = ckpt['epoch'] + 1
    if config['model']['is_finetune'] is False:
        best_valid_loss = ckpt['loss']
        global_iter_train = ckpt['global_train_iter']
        global_iter_valid = ckpt['global_valid_iter']
        # hvd.broadcast_object(start_epoch, root_rank=0)
    else:
        start_epoch = 0
    optimizer = optimizer.load_state_dict(ckpt['optimizer'])
    # scheduler_for_lr = ckpt['scheduler']

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# print out
print("optimizer : " + str(optimizer))
if scheduler_for_lr is None:
    print("lr_scheduler : None")
elif config['params']['lr_multistep'] != 'None':
    tmp_str = "lr_scheduler : [milestones: "
    for milestone in scheduler_for_lr.milestones:
        tmp_str = tmp_str + str(milestone) + ', '
    tmp_str += ' gamma: '
    tmp_str += str(scheduler_for_lr.gamma)
    tmp_str += ']'
    print(tmp_str)
print("Size of batch : " + str(train_loader.batch_size))
print("transform : " + str(train_transforms))
print("num. train data : " + str(len(train_dataset)))
print("num. valid data : " + str(len(valid_dataset)))
print("num_classes : " + str(num_classes))
print("classes : " + str(target_classes))

utils.print_config(config)

# input("Press any key to continue..")


# Training
def train(epoch):
    # eps=1e-9
    net.train()
    train_loss = 0.

    global global_iter_train

    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    with torch.set_grad_enabled(True):
        # with autograd.detect_anomaly():
        # for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(train_loader):
        for batch_idx, (inputs, loc_targets, cls_targets, paths) in enumerate(train_loader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)
            # mask_targets = mask_targets.to(device)

            # loc_preds, cls_preds, mask_preds = net(inputs)
            loc_preds, cls_preds = net(inputs)

            # loc_loss, cls_loss, mask_loss, num_matched_anchors = \
            #     criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
            loc_loss, cls_loss, num_matched_anchors = \
                criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            if num_matched_anchors == 0:
                print('No matched anchor')
                continue
            else:
                num_matched_anchors = float(num_matched_anchors)
                # loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
                loss = (loc_loss + cls_loss) / num_matched_anchors

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print('[Train#%d] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f | '
            #       'train_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
            #       % (hvd.rank(), epoch, batch_idx, loc_loss.item(), cls_loss.item(), mask_loss.item(),
            #          loss.item(), train_loss/(batch_idx+1), num_matched_anchors))
            print('[Train#%d] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | '
                  'train_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
                  % (hvd.rank(), epoch, batch_idx, loc_loss.item(), cls_loss.item(),
                     loss.item(), train_loss/(batch_idx+1), num_matched_anchors))

            if hvd.rank() == 0:
                summary_writer.add_scalar('train/loc_loss', loc_loss.item(), global_iter_train)
                summary_writer.add_scalar('train/cls_loss', cls_loss.item(), global_iter_train)
                # summary_writer.add_scalar('train/mask_loss', mask_loss.item(), global_iter_train)
                summary_writer.add_scalar('train/train_loss', loss.item(), global_iter_train)
                global_iter_train += 1

                if config['params']['lr_multistep'] != 'None':
                    scheduler_for_lr.step()

        if hvd.rank() == 0:
            print('[Train] Saving..')
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'lr': config['params']['lr'],
                'anchors': num_anchors,
                'classes': config['params']['classes'],
                'global_train_iter': global_iter_train,
                'global_valid_iter': global_iter_valid,
                'optimizer': optimizer.state_dict()
            }
            # torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')
            torch.save(state, os.path.join(config['model']['exp_path'], 'latest.pth'))


# Valid
def valid(epoch):
    # eps = 1e-9
    net.eval()
    valid_loss = 0.
    avg_valid_loss = 0.
    is_saved = False

    global best_valid_loss
    global global_iter_valid
    global global_iter_train

    with torch.set_grad_enabled(False):
        # for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(valid_loader):
        for batch_idx, (inputs, loc_targets, cls_targets, paths) in enumerate(valid_loader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)
            # mask_targets = mask_targets.to(device)

            # loc_preds, cls_preds, mask_preds = net(inputs)
            loc_preds, cls_preds = net(inputs)

            # loc_loss, cls_loss, mask_loss, num_matched_anchors = \
            #     criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
            loc_loss, cls_loss, num_matched_anchors = \
                criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            if num_matched_anchors == 0:
                print('No matched anchor')
                continue
            num_matched_anchors = float(num_matched_anchors)
            # loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
            loss = ((loc_loss + cls_loss) / num_matched_anchors)
            valid_loss += loss.item()
            avg_valid_loss = valid_loss / (batch_idx + 1)
            # print('[Valid#%d] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f | '
            #       'valid_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
            #     % (hvd.rank(), epoch, batch_idx, loc_loss.item(), cls_loss.item(), mask_loss.item(),
            #        loss.item(), avg_valid_loss, num_matched_anchors))
            print('[Valid#%d] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | '
                  'valid_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
                % (hvd.rank(), epoch, batch_idx, loc_loss.item(), cls_loss.item(),
                   loss.item(), avg_valid_loss, num_matched_anchors))

            if hvd.rank() == 0:
                summary_writer.add_scalar('valid/loc_loss', loc_loss.item(), global_iter_valid)
                summary_writer.add_scalar('valid/cls_loss', cls_loss.item(), global_iter_valid)
                # summary_writer.add_scalar('valid/mask_loss', mask_loss.item(), global_iter_valid)
                summary_writer.add_scalar('valid/valid_loss', loss.item(), global_iter_valid)
                global_iter_valid += 1

    # check whether better model or not
    all_avg_valid_loss = hvd.allreduce(torch.tensor(avg_valid_loss), name='all_avg_loss')

    if hvd.rank() == 0:
        if all_avg_valid_loss.item() < best_valid_loss:
            best_valid_loss = all_avg_valid_loss.item()
            is_saved = True

        if is_saved is True:
            print('[Valid] Saving..')
            state = {
                'net': net.state_dict(),
                'loss': best_valid_loss,
                'epoch': epoch,
                'lr': config['params']['lr'],
                'anchors': num_anchors,
                'classes': config['params']['classes'],
                'global_train_iter': global_iter_train,
                'global_valid_iter': global_iter_valid,
                'optimizer': optimizer.state_dict()
            }
            # torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')
            torch.save(state, os.path.join(config['model']['exp_path'], 'best.pth'))


if __name__ == '__main__':
    for epoch in range(start_epoch, config['params']['epoch'] + 1, 1):
        train(epoch)
        valid(epoch)
    if hvd.rank() == 0:
        summary_writer.close()

    print("best valid loss : " + str(best_valid_loss))
