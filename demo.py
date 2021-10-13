# python
import os
import argparse
import math
import numpy as np
import shutil
import cv2
from tqdm import tqdm

# pytorch
import torch
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# user-defined
from models.retinanet import load_model
import utils
from encoder import DataEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
parser.add_argument('--imgs', type=str, required=True, help='path of image files')
opt = parser.parse_args()

config = utils.get_config(opt.config)

# cls_th = float(config['params']['cls_threshold'])
cls_th = 0.5
nms_th = 0.5

output_dir = os.path.join(config['model']['exp_path'], 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

''' cuda '''
if torch.cuda.is_available() and not config['cuda']['using_cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

cuda_str = 'cuda:' + str(config['cuda']['gpu_id'])
device = torch.device(cuda_str if config['cuda']['using_cuda'] else "cpu")

''' target class and target image size'''
target_classes = utils.read_txt(config['params']['classes'])
num_classes = len(target_classes)

is_resized = True
if config['inference']['resized'] is True:
    print('Do normalize to image size')
else:
    is_resized = False
    print('Do not normalize to image size')

if isinstance(config['inference']['image_size'], str):
    img_size = config['inference']['image_size'].split('x')
    img_size = (int(img_size[0]), int(img_size[1])) # rows x cols
    print('Image size: ' + config['inference']['image_size'])
else:
    print(config['inference']['image_size'])
    raise ValueError('Check out image size.')

''' get random box color '''
bbox_colormap = utils._get_rand_bbox_colormap(class_names=target_classes)

''' transforms '''
# bbox_params = A.BboxParams(format='pascal_voc', min_visibility=0.3)
valid_transforms = A.Compose([
    A.Resize(height=img_size[0], width=img_size[1], p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
], p=1.0)


''' load network'''
best_ckpt_path = os.path.join(config['model']['exp_path'], 'best.pth')
print(best_ckpt_path)
ckpt = torch.load(best_ckpt_path, map_location=device)

net = load_model(num_classes=num_classes,
                 num_anchors=ckpt['anchors'],
                 basenet=config['params']['base'],
                 is_pretrained_base=False)
net = net.to(device)
net.eval()

weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
print(missing_keys)

''' load box encoder and images '''
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

''' make output dir '''
img_dir_name = os.path.split(opt.imgs)[-1]
result_dir = os.path.join(config['model']['exp_path'], 'results', img_dir_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

''' do inference'''
with torch.set_grad_enabled(False):
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        ori_rows = img.shape[0]
        ori_cols = img.shape[1]

        # render_img = img
        ''' no resize the target image'''
        if is_resized is False:
            img_size = (int(ori_rows / 4), int(ori_cols / 4))

            list_boxes = list()
            list_labels = list()
            list_scores = list()

            margin = [int(img_size[0] * 0.20), int(img_size[1] * 0.20)]

            num_row_iter = int(math.ceil((ori_rows - img_size[0]) / (img_size[0] - margin[0])))
            num_col_iter = int(math.ceil((ori_cols - img_size[1]) / (img_size[1] - margin[1])))

            for rows_idx in range(num_row_iter + 1):
                for cols_idx in range(num_col_iter + 1):
                    crop_ymin = rows_idx * img_size[0] - (margin[0] * rows_idx)
                    crop_xmin = cols_idx * img_size[1] - (margin[1] * cols_idx)

                    if crop_ymin < 0:
                        crop_ymin = 0.0
                    if crop_xmin < 0:
                        crop_xmin = 0.0

                    crop_ymax = crop_ymin + img_size[0]
                    crop_xmax = crop_xmin + img_size[1]

                    if crop_ymax >= ori_rows:
                        crop_ymin = crop_ymin - (crop_ymax - (ori_rows-1))
                        crop_ymax = crop_ymin + img_size[0]
                    if crop_xmax >= ori_cols:
                        crop_xmin = crop_xmin - (crop_xmax - (ori_cols - 1))
                        crop_xmax = crop_xmin + img_size[1]

                    crop_img = img[int(crop_ymin):int(crop_ymax), int(crop_xmin):int(crop_xmax)]

                    # cv2.rectangle(render_img, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), (0, 255, 0))
                    # cv2.imshow('test', render_img)
                    # cv2.waitKey(0)

                    augmented = valid_transforms(image=crop_img)
                    img = augmented['image']
                    x = img.unsqueeze(0)
                    x = x.to(device)

                    # loc_preds, cls_preds, mask_preds = net(x)
                    loc_preds, cls_preds = net(x)

                    boxes, labels, scores = data_encoder.decode(loc_preds=loc_preds.squeeze(),
                                                                cls_preds=cls_preds.squeeze(),
                                                                input_size=(img_size[1], img_size[0]),
                                                                cls_threshold=cls_th,
                                                                top_k=-1)
                    if isinstance(boxes, list):
                        continue

                    for box in boxes:
                        box[0] += crop_xmin
                        box[1] += crop_ymin
                        box[2] += crop_xmin
                        box[3] += crop_ymin

                    list_boxes.append(boxes)
                    list_labels.append(labels)
                    list_scores.append(scores)

            if len(list_boxes) > 0:
                boxes = torch.cat(list_boxes, dim=0)
                labels = torch.cat(list_labels, dim=0)
                scores = torch.cat(list_scores, dim=0)
            else:
                boxes = list()
                labels = list()
                scores = list()
        else:
            ''' resize the target image '''
            resized_img = cv2.resize(img, (img_size[1], img_size[0]))
            augmented = valid_transforms(image=resized_img)
            img = augmented['image']
            x = img.unsqueeze(0)
            x = x.to(device)

            # loc_preds, cls_preds, mask_preds = net(x)
            loc_preds, cls_preds = net(x)

            boxes, labels, scores = data_encoder.decode(loc_preds=loc_preds.squeeze(),
                                                        cls_preds=cls_preds.squeeze(),
                                                        input_size=(img_size[1], img_size[0]),
                                                        cls_threshold=cls_th,
                                                        top_k=config['inference']['top_k'])

        ''' do nms '''
        if len(boxes) > 0:
            # nms mode = 0: soft-nms(liner), 1: soft-nms(gaussian), 2: hard-nms
            keep = utils.box_nms(boxes, scores, nms_threshold=nms_th, mode=2)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        ''' write output file'''
        if is_resized is False:
            utils._write_results(dir_path=result_dir,
                                 img_path=img_path,
                                 boxes=boxes,
                                 scores=scores,
                                 labels=labels,
                                 class_idx_map=class_idx_map,
                                 input_size=(ori_rows, ori_cols),
                                 bbox_colormap=bbox_colormap)
        else:
            utils._write_results(dir_path=result_dir,
                                 img_path=img_path,
                                 boxes=boxes,
                                 scores=scores,
                                 labels=labels,
                                 class_idx_map=class_idx_map,
                                 input_size=img_size,
                                 bbox_colormap=bbox_colormap)


