# python
import os
import argparse
import random
import numpy as np
import shutil
import cv2
from tqdm import tqdm

# pytorch
import torch
import torchvision.transforms as transforms

# user-defined
from models.retinanet import load_model
import utils
from encoder import DataEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
parser.add_argument('--imgs', type=str, required=True, help='path of image files')
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

is_resized = True
target_classes = config['hyperparameters']['classes'].split('|')
if isinstance(config['hyperparameters']['image_size'], str) == True:
    img_size = config['hyperparameters']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1])) # rows x cols
    print('Image size(normalization): ' + config['hyperparameters']['image_size'])
else:
    img_size = None
    is_resized = False
    print('Do not normalize to image size')

num_classes = len(target_classes)

ckpt = torch.load(os.path.join(config['model']['exp_path'], 'best.pth'), map_location=device)

net = load_model(num_classes=num_classes,
                 num_anchors=ckpt['anchors'],
                 basenet=config['hyperparameters']['base'],
                 is_pretrained_base=False)
net = net.to(device)
net.eval()

weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
print(missing_keys)

data_encoder = DataEncoder()

class_idx_map = dict()
for idx in range(0, num_classes):
    class_idx_map[idx + 1] = target_classes[idx]

img_paths = list()
for (path, _, files) in os.walk(opt.imgs):
    for file in files:
        ext = os.path.splitext(file)[-1]

        if ext == '.jpg':
            img_paths.append(os.path.join(path, file))

img_dir_name = os.path.split(opt.imgs)[-1]
result_dir = os.path.join(config['model']['exp_path'], 'results', img_dir_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

with torch.set_grad_enabled(False):
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        ori_rows = img.shape[0]
        ori_cols = img.shape[1]
        if is_resized is False:
            img_size = (ori_rows, ori_cols)
            if ori_cols > 3840:
                ratio = ori_cols / ori_rows
                img_size = (int(3840 * (1 / ratio)), 3840)
                print(img_size)
        resized_img = cv2.resize(img, (img_size[1], img_size[0]))
        x = transform(resized_img)
        x = x.unsqueeze(0)
        x = x.to(device)

        loc_preds, cls_preds, mask_preds = net(x)

        boxes, labels, scores = data_encoder.decode(loc_preds=loc_preds.squeeze(),
                                                    cls_preds=cls_preds.squeeze(),
                                                    input_size=(img_size[1], img_size[0]),
                                                    cls_threshold=cls_th,
                                                    nms_threshold=nms_th)

        utils._write_results(result_dir, img_path, boxes, scores, labels, class_idx_map, img_size)



