'''Some helper functions for PyTorch.'''

import torch
import yaml
import os
import collections
import cv2
import numpy as np


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0, x, dtype=torch.float32)
    b = torch.arange(0, y, dtype=torch.float32)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2, b-a], 1)
    return torch.cat([a-(b/2), a+(b/2)], 1)


def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def sort_with_indices(values, indices):
    num_elem = values.numel()

    '''bubble sort'''
    for current in range(0, num_elem, 1):
        for next_idx in range(current+1, num_elem, 1):
            if values[next_idx] > values[current]:
                tmp_value = values[current].item()
                tmp_idx = indices[current].item()
                values[current] = values[next_idx].item()
                indices[current] = indices[next_idx].item()
                values[next_idx] = tmp_value
                indices[next_idx] = tmp_idx


def box_nms(bboxes, scores, nms_threshold, mode, ovr_mode='union', soft_threshold=0.01):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      nms_threshold: (float) overlap threshold.
      cls_threshold: (float) classification threshold.
      mode: (str) 'soft' or 'hard'.
      ovr_mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.
    '''
    sigma = 0.5

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2-x1+1) * (y2-y1+1)
    ordered_score, ordered_idx = scores.sort(0, descending=True)

    keep = []
    while ordered_idx.numel() > 0:
        if ordered_idx.numel() == 1:
            max_idx = ordered_idx.item()
            keep.append(max_idx)
            break

        max_idx = ordered_idx[0]
        keep.append(max_idx)

        xx1 = x1[ordered_idx[1:]].clamp(min=x1[max_idx].item())
        yy1 = y1[ordered_idx[1:]].clamp(min=y1[max_idx].item())
        xx2 = x2[ordered_idx[1:]].clamp(max=x2[max_idx].item())
        yy2 = y2[ordered_idx[1:]].clamp(max=y2[max_idx].item())

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if ovr_mode == 'union':
            ovr = inter / (areas[max_idx] + areas[ordered_idx[1:]] - inter)
        elif ovr_mode == 'min':
            ovr = inter / areas[ordered_idx[1:]].clamp(max=areas[max_idx])
        else:
            raise TypeError('Unknown nms mode: %s.' % ovr_mode)

        weights = torch.zeros_like(ovr)
        if mode == 0:
            # soft-nms(linear)
            ovr_idx = ovr > nms_threshold
            non_ovr_idx = torch.logical_not(ovr_idx)

            weights[ovr_idx] = 1.0 - ovr[ovr_idx]
            weights[non_ovr_idx] = 1.0
        elif mode == 1:
            # soft-nms(gaussian)
            weights = torch.exp(-1. * (torch.pow(ovr, 2) / sigma))
        else:
            # hard-nms
            ovr_idx = ovr > nms_threshold
            non_ovr_idx = torch.logical_not(ovr_idx)

            weights[ovr_idx] = 0.0
            weights[non_ovr_idx] = 1.0

        ordered_idx = ordered_idx[1:]
        ordered_score = weights * ordered_score[1:]
        ids = (ordered_score > soft_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        ordered_idx = ordered_idx[ids]
        ordered_score = ordered_score[ids]

        if mode < 2:
            # sort for soft-nms
            sort_with_indices(ordered_score, ordered_idx)
    return torch.LongTensor(keep)


def softmax(x):
    '''Softmax along a specific dimension.

    Args:
      x: (tensor) input tensor, sized [N,D].

    Returns:
      (tensor) softmaxed tensor, sized [N,D].
    '''
    xmax, _ = x.max(1)
    x_shift = x - xmax.view(-1,1)
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(1).view(-1, 1)


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    num_of_data = labels.size(0)
    dims = num_classes
    y = torch.zeros([num_of_data, dims], dtype=torch.float32, device=labels.device)
    y[torch.arange(0, num_of_data).long(), labels] = 1

    return y


def get_config(conf):
    with open(conf, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def print_config(conf):
    print(yaml.dump(conf, default_flow_style=False, default_style=''))


def get_best_model(dir_path):
    ckpt_file = dict()
    minimum_loss = float('inf')
    minimum_file = ''

    for (path, dirs, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.pth':
                load_pth = torch.load(os.path.join(path, filename), map_location='cpu')
                valid_loss = load_pth['loss']

                ckpt_idx = filename
                ckpt_idx = int(ckpt_idx.split("-")[-1].split(".")[0])

                ckpt_file[ckpt_idx] = valid_loss

                if valid_loss < minimum_loss:
                    minimum_loss = valid_loss
                    minimum_file = filename

    for idx in ckpt_file:
        print("ckpt-" + str(idx) + " " + str(ckpt_file[idx]))

    if minimum_file == '':
        return None

    return os.path.join(dir_path, minimum_file)

def _load_weights(weights_dict):
    key, value = list(weights_dict.items())[0]

    trained_data_parallel = False
    if key[:7] == 'module.':
        trained_data_parallel = True

    if trained_data_parallel is True:
        new_weights = collections.OrderedDict()
        for old_key in weights_dict:
            new_key = old_key[7:]
            new_weights[new_key] = weights_dict[old_key]
    else:
        new_weights = weights_dict

    return new_weights

def _get_rand_bbox_colormap(class_names):
    color_dict = dict()
    for class_name in class_names:
        np_rand_vals = np.random.choice(range(256), size=3)
        rand_color = (int(np_rand_vals[0]), int(np_rand_vals[1]), int(np_rand_vals[2]))
        color_dict[class_name] = rand_color

    return color_dict

def _get_box_color(class_name, rand_colormap):
    return rand_colormap[class_name]


def _get_class_name(class_idx, class_idx_map):
    return class_idx_map[class_idx]


def _draw_rects(img, boxes, scores, labels, class_idx_map, ws, hs, bbox_colormap):
    for box_idx, box in enumerate(boxes):
        for iter_idx in range(len(box)):
            if box[iter_idx] < 0:
                box[iter_idx] = 0
        ''' get box '''
        pt1 = (int(box[0] * ws), int(box[1] * hs))
        pt2 = (int(box[2] * ws), int(box[3] * hs))
        class_name = _get_class_name(labels[box_idx]+1, class_idx_map)
        score = float(scores[box_idx])
        out_text = class_name + ':' + format(score, ".2f")
        box_color = _get_box_color(class_name, bbox_colormap)

        ''' draw rect '''
        roi_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        fill_rect = np.ones(roi_img.shape) * box_color
        fill_rect = fill_rect.astype(np.uint8)
        trans_rect = cv2.addWeighted(roi_img, 0.5, fill_rect, 0.5, 0)
        img[pt1[1]:pt2[1], pt1[0]:pt2[0]] = trans_rect
        cv2.rectangle(img, pt1, pt2, box_color, 1)

        ''' write font '''
        t_size = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        pt2 = pt1[0] + (t_size[0] + 3), pt1[1] - (t_size[1] + 4)
        cv2.rectangle(img, pt1, pt2, box_color, -1)
        cv2.putText(img, out_text, (pt1[0], pt1[1] - (t_size[1] - 7)), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

def _write_txt(out_path, boxes, scores, labels, class_idx_map, ws, hs):
    f_out = open(out_path, 'w')

    for box_idx, box in enumerate(boxes):
        pt1 = (int(box[0] * ws), int(box[1] * hs))
        pt2 = (int(box[2] * ws), int(box[3] * hs))
        class_name = _get_class_name(labels[box_idx] + 1, class_idx_map)
        score = scores[box_idx]

        out_txt = str(class_name) + '\t' + \
                  str(pt1[0]) + '\t' + str(pt1[1]) + '\t' + str(pt2[0]) + '\t' + str(pt2[1]) + '\t' \
                  + str(score) + '\n'
        f_out.write(out_txt)

    f_out.close()


def _write_results(dir_path, img_path, boxes, scores, labels, class_idx_map, input_size, bbox_colormap):
    if not isinstance(boxes, list):
        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

    image_name = os.path.basename(img_path)
    image_ext = os.path.splitext(img_path)[-1]
    image_name = image_name.replace(image_ext, '')

    img = cv2.imread(img_path)
    resized_rows = input_size[0]
    resized_cols = input_size[1]
    ori_rows = img.shape[0]
    ori_cols = img.shape[1]
    ws = ori_cols / resized_cols
    hs = ori_rows / resized_rows
    _draw_rects(img, boxes=boxes, scores=scores, labels=labels, class_idx_map=class_idx_map, ws=ws, hs=hs,
                bbox_colormap=bbox_colormap)

    img_out = os.path.join(dir_path, image_name + image_ext)
    cv2.imwrite(img_out, img)
    _write_txt(os.path.join(dir_path, image_name + '.txt'), boxes=boxes, scores=scores, labels=labels,
               class_idx_map=class_idx_map, ws=ws, hs=hs)

def read_txt(txt_path):
    f_read = open(txt_path, 'r')
    lines = f_read.readlines()

    out = list()
    for line in lines:
        out.append(line.rstrip())

    return out
