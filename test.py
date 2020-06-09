# python
import os
import argparse
import random
import numpy as np
import shutil
import sys

# pytorch
import torch
import torchvision.transforms as transforms

# user-defined
from models.retinanet import load_model
from datagen import jsonDataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)

cls_th = float(config['hyperparameters']['cls_threshold'])
nms_th = 0.5

output_dir = os.path.join(config['model']['exp_path'], 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# cuda
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

target_classes = config['hyperparameters']['classes'].split('|')
img_size = config['hyperparameters']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))   # rows x cols
num_classes = len(target_classes)
best_ckpt_path = os.path.join(config['model']['exp_path'], 'best.pth')
print(best_ckpt_path)
ckpt = torch.load(best_ckpt_path, map_location=device)

net = load_model(num_classes=num_classes,
                 num_anchors=ckpt['anchors'],
                 basenet=config['hyperparameters']['base'],
                 is_pretrained_base=False)
net = net.to(device)
net.eval()

weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
print(missing_keys)

dataset_dict = config['data']
for dataset_name in dataset_dict:
    data_path = dataset_dict[dataset_name]
    if data_path.split(' ')[-1] == 'notest':
        continue
    print(dataset_name)
    result_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    dataset = jsonDataset(path=data_path, classes=target_classes, transform=transform, input_image_size=img_size,
                          num_crops=-1)
    num_data = len(dataset)
    assert dataset
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['hyperparameters']['batch_size'],
        shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn)

    with torch.set_grad_enabled(False):
        for batch_idx, (inputs, loc_targets, cls_targets, mask_targets, paths) in enumerate(data_loader):
            sys.stdout.write('\r' + str(batch_idx * config['hyperparameters']['batch_size']) + ' / ' + str(num_data))
            inputs = inputs.to(device)
            loc_preds, cls_preds, mask_preds = net(inputs)
            num_batch = loc_preds.shape[0]

            for iter_batch in range(num_batch):
                boxes, labels, scores = dataset.data_encoder.decode(loc_preds=loc_preds[iter_batch],
                                                                    cls_preds=cls_preds[iter_batch],
                                                                    input_size=(img_size[1], img_size[0]),
                                                                    cls_threshold=cls_th,
                                                                    nms_threshold=nms_th)

                utils._write_results(result_dir, paths[iter_batch], boxes, scores, labels, dataset.class_idx_map,
                                     img_size)
    print()







