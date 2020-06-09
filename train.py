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

# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter

# user-defined
from loss import FocalLoss
from models.retinanet import load_model
from datagen import jsonDataset
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)
start_epoch = 0  # start from epoch 0 or last epoch

# make output folder
if not os.path.exists(config['model']['exp_path']):
    os.mkdir(config['model']['exp_path'])

if not os.path.exists(os.path.join(config['model']['exp_path'], 'config.yaml')):
    shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))
else:
    os.remove(os.path.join(config['model']['exp_path'], 'config.yaml'))
    shutil.copy(opt.config, os.path.join(config['model']['exp_path'], 'config.yaml'))

# set random seed
random.seed(config['hyperparameters']['random_seed'])
np.random.seed(config['hyperparameters']['random_seed'])
torch.manual_seed(config['hyperparameters']['random_seed'])

# variables
best_valid_loss = float('inf')

global_iter_train = 0
global_iter_valid = 0

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

is_data_parallel = False
if isinstance(config['cuda']['gpu_id'], list):
    is_data_parallel = True
    cuda_str = 'cuda:' + str(config['cuda']['gpu_id'][0])
elif isinstance(config['cuda']['gpu_id'], int):
    cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
else:
    raise ValueError('Check out gpu id in config')

device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

# tensorboard
summary_writer = SummaryWriter(os.path.join(config['model']['exp_path'], 'log'))

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor()
])

target_classes = config['hyperparameters']['classes'].split('|')
img_size = config['hyperparameters']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes,
                            transform=transform,
                            input_image_size=img_size,
                            num_crops=config['hyperparameters']['num_crops'],
                            do_aug=config['hyperparameters']['do_aug'])

valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes,
                            transform=transform,
                            input_image_size=img_size,
                            num_crops=config['hyperparameters']['num_crops'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['hyperparameters']['batch_size'],
    shuffle=True, num_workers=config['hyperparameters']['data_worker'],
    collate_fn=train_dataset.collate_fn,
    pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config['hyperparameters']['batch_size'],
    shuffle=False, num_workers=config['hyperparameters']['data_worker'],
    collate_fn=valid_dataset.collate_fn,
    pin_memory=True)

assert train_dataset
assert valid_dataset

# Model
num_classes = len(target_classes)
num_anchors = train_dataset.data_encoder.num_anchors

net = load_model(num_classes=num_classes,
                 num_anchors=num_anchors,
                 basenet=config['hyperparameters']['base'],
                 is_pretrained_base=config['hyperparameters']['pre_base'])
net = net.to(device)

# print out net
num_parameters = 0.
for param in net.parameters():
    sizes = param.size()

    num_layer_param = 1.
    for size in sizes:
        num_layer_param *= size
    num_parameters += num_layer_param
print(net)
print("num. of parameters : " + str(num_parameters))

# set data parallel
if is_data_parallel is True:
    net = torch.nn.DataParallel(module=net, device_ids=config['cuda']['gpu_id'])

# loss
criterion = FocalLoss(num_classes=num_classes)

# optimizer
if config['hyperparameters']['optimizer'] == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=float(config['hyperparameters']['lr']), momentum=0.9, weight_decay=5e-4)
elif config['hyperparameters']['optimizer'] == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=float(config['hyperparameters']['lr']))
else:
    raise ValueError('not supported optimizer')

# set lr scheduler
if config['hyperparameters']['lr_patience'] > 0:
    scheduler_for_lr = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                      patience=config['hyperparameters']['lr_patience'], verbose=True)
else:
    scheduler_for_lr = None

if config['hyperparameters']['lr_multistep'] != 'None':
    milestones = config['hyperparameters']['lr_multistep']
    milestones = milestones.split(', ')
    for iter_milestone in range(len(milestones)):
        milestones[iter_milestone] = int(milestones[iter_milestone])
    scheduler_for_lr = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
else:
    scheduler_for_lr = None

# set pre-trained
if config['model']['model_path'] != 'None':
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
    else:
        start_epoch = 0
    optimizer = ckpt['optimizer']
    # scheduler_for_lr = ckpt['scheduler']

# print out
print("optimizer : " + str(optimizer))
if scheduler_for_lr is None:
    print("lr_scheduler : None")
elif config['hyperparameters']['lr_patience'] > 0:
    print("lr_scheduler : [patience: " + str(scheduler_for_lr.patience) +
          ", gamma: " + str(scheduler_for_lr.factor) +"]")
elif config['hyperparameters']['lr_multistep'] != 'None':
    tmp_str = "lr_scheduler : [milestones: "
    for milestone in scheduler_for_lr.milestones:
        tmp_str = tmp_str + str(milestone) + ', '
    tmp_str += ' gamma: '
    tmp_str += str(scheduler_for_lr.gamma)
    tmp_str += ']'
    print(tmp_str)
print("Size of batch : " + str(train_loader.batch_size))
print("transform : " + str(transform))
print("num. train data : " + str(len(train_dataset)))
print("num. valid data : " + str(len(valid_dataset)))
print("num_classes : " + str(num_classes))
print("classes : " + str(target_classes))

utils.print_config(config)

input("Press any key to continue..")


# Training
def train(epoch):
    # eps=1e-9
    net.train()
    train_loss = 0.

    global global_iter_train

    with torch.set_grad_enabled(True):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(train_loader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)
            mask_targets = mask_targets.to(device)

            optimizer.zero_grad()
            loc_preds, cls_preds, mask_preds = net(inputs)

            loc_loss, cls_loss, mask_loss, num_matched_anchors = \
                criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
            if num_matched_anchors == 0:
                print('No matched anchor')
                loss = loc_loss + cls_loss + mask_loss
                loss.backward()
                continue
            num_matched_anchors = float(num_matched_anchors)
            loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('[Train] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f | '
                  'train_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
                  % (epoch, batch_idx, loc_loss.item(), cls_loss.item(), mask_loss.item(),
                     loss.item(), train_loss/(batch_idx+1), num_matched_anchors))

            summary_writer.add_scalar('train/loc_loss', loc_loss.item(), global_iter_train)
            summary_writer.add_scalar('train/cls_loss', cls_loss.item(), global_iter_train)
            summary_writer.add_scalar('train/mask_loss', mask_loss.item(), global_iter_train)
            summary_writer.add_scalar('train/train_loss', loss.item(), global_iter_train)
            global_iter_train += 1

            if config['hyperparameters']['lr_multistep'] != 'None':
                scheduler_for_lr.step()

        print('[Train] Saving..')
        state = {
            'net': net.state_dict(),
            'loss': best_valid_loss,
            'epoch': epoch,
            'lr': config['hyperparameters']['lr'],
            'batch': config['hyperparameters']['batch_size'],
            'anchors': num_anchors,
            'classes': config['hyperparameters']['classes'],
            'global_train_iter': global_iter_train,
            'global_valid_iter': global_iter_valid,
            'optimizer': optimizer
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
        for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(valid_loader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)
            mask_targets = mask_targets.to(device)

            loc_preds, cls_preds, mask_preds = net(inputs)

            loc_loss, cls_loss, mask_loss, num_matched_anchors = \
                criterion(loc_preds, loc_targets, cls_preds, cls_targets, mask_preds, mask_targets)
            if num_matched_anchors == 0:
                print('No matched anchor')
                continue
            num_matched_anchors = float(num_matched_anchors)
            loss = ((loc_loss + cls_loss) / num_matched_anchors) + mask_loss
            valid_loss += loss.item()
            avg_valid_loss = valid_loss / (batch_idx + 1)
            print('[Valid] epoch: %3d | iter: %4d | loc_loss: %.3f | cls_loss: %.3f | mask_loss: %.3f | '
                  'valid_loss: %.3f | avg_loss: %.3f | matched_anchors: %d'
                % (epoch, batch_idx, loc_loss.item(), cls_loss.item(), mask_loss.item(),
                   loss.item(), avg_valid_loss, num_matched_anchors))

            summary_writer.add_scalar('valid/loc_loss', loc_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid/cls_loss', cls_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid/mask_loss', mask_loss.item(), global_iter_valid)
            summary_writer.add_scalar('valid/valid_loss', loss.item(), global_iter_valid)
            global_iter_valid += 1

    if config['hyperparameters']['lr_patience'] > 0:
        scheduler_for_lr.step(avg_valid_loss)

    # check whether better model or not
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        is_saved = True

    if is_saved is True:
        print('[Valid] Saving..')
        state = {
            'net': net.state_dict(),
            'loss': best_valid_loss,
            'epoch': epoch,
            'lr': config['hyperparameters']['lr'],
            'batch': config['hyperparameters']['batch_size'],
            'anchors': num_anchors,
            'classes': config['hyperparameters']['classes'],
            'global_train_iter': global_iter_train,
            'global_valid_iter': global_iter_valid,
            'optimizer': optimizer
        }
        # torch.save(state, config['model']['exp_path'] + '/ckpt-' + str(epoch) + '.pth')
        torch.save(state, os.path.join(config['model']['exp_path'], 'best.pth'))


if __name__ == '__main__':
    for epoch in range(start_epoch, config['hyperparameters']['epoch'], 1):
        train(epoch)
        valid(epoch)
    summary_writer.close()

    print("best valid loss : " + str(best_valid_loss))
