import argparse
import os
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import mlflow
from retina import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of ex config')
parser.add_argument('--model', type=str, required=True, help='model architecture')
parser.add_argument('--view_result', action='store_true', help='whether see graph or not')
parser.add_argument('--view_image', action='store_true', help='whether see image or not')
parser.add_argument('--mlflow', action='store_true', help='store results to mlflow')
opt = parser.parse_args()

FILE_EXT = '.txt'
IOU_THRESHOLD = 0.5
mlflow.set_tracking_uri(uri='/home/hotaekhan/mlruns')


def _get_iou(box1, box2):
    lt = [max(box1[0], box2[0]), max(box1[1], box2[1])]
    rb = [min(box1[2], box2[2]), min(box1[3], box2[3])]

    width = rb[0] - lt[0]
    height = rb[1] - lt[1]
    inter = width * height

    if width < 0 or height < 0:
        return 0.0

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area_box1+area_box2-inter)

    return iou


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


if __name__ == '__main__':
    config = utils.get_config(opt.config)
    target_datasets = config['data']

    if opt.mlflow is True:
        mlflow.set_experiment(config['model']['exp_name'])

    target_classes = config['params']['classes'].split('|')
    num_classes = len(target_classes)
    class_dict = dict()
    class_idx_dict = dict()
    for idx in range(0, num_classes):
        class_dict[target_classes[idx]] = 0
        class_idx_dict[idx] = target_classes[idx]

    for dataset_name in target_datasets:
        if target_datasets[dataset_name].split(' ')[-1] == 'notest':
            continue
        not_exist_class = 0
        # 1. read gt
        '''
        gt data structure
        gt(dict) = key is image id
        gt[key] = bboxes
        bboxes(list) : list of bbox
        bbox(dict) = keys are box coordinate, class, is_matched
        '''
        print('[' + str(dataset_name) + '] read gt boxes..')
        fp_read = open(os.path.join(opt.model, target_datasets[dataset_name]), 'r')
        gt = json.load(fp_read)

        gt_dict = dict()
        gt_boxes_per_class = dict()
        path_dict = dict()

        for gt_key in tqdm(gt):
            gt_data = gt[gt_key][0]
            num_boxes = len(gt_data['labels'])

            ext = os.path.splitext(gt_data['image_path'])[-1]
            image_id = os.path.basename(gt_data['image_path']).replace(ext, '')

            gt_bboxes = list()
            for iter_box in range(0, num_boxes):
                gt_bbox = dict()

                xmin = gt_data['boxes'][iter_box][0]
                ymin = gt_data['boxes'][iter_box][1]
                xmax = gt_data['boxes'][iter_box][2]
                ymax = gt_data['boxes'][iter_box][3]
                class_name = gt_data['labels'][iter_box][0]

                gt_bbox['box'] = [xmin, ymin, xmax, ymax]
                gt_bbox['class'] = class_name
                gt_bbox['matched'] = False
                gt_bbox['image_path'] = gt_data['image_path']
                gt_bboxes.append(gt_bbox)

                if class_name not in gt_boxes_per_class:
                    gt_boxes_per_class[class_name] = 1
                else:
                    gt_boxes_per_class[class_name] += 1

            gt_dict[image_id] = gt_bboxes
            path_dict[image_id] = gt_data['image_path']

        # 2. read predicted box
        '''
        pred box structure
        pred_box(dict) = key is class name
        pred_box['class_name'] : list of box
        bbox(dict) = keys are box coordinate, class, confidence, image_id
        list(bbox) is sorted by confidence in descending order.
        '''
        print('read pred boxes..')
        pred_files = list()
        pred_path = os.path.join(config['model']['exp_path'], 'results', dataset_name)

        if not os.path.exists(pred_path):
            raise FileExistsError(str(pred_path) + ' is not exist.')

        for (path, _, files) in os.walk(pred_path):
            for file in files:
                ext = os.path.splitext(file)[-1]

                if ext == FILE_EXT:
                    pred_files.append(os.path.join(path, file))

        pred_dict = dict()
        for class_name in class_dict:
            pred_dict[class_name] = list()

        for pred_file in tqdm(pred_files):
            image_id = os.path.basename(pred_file).replace(FILE_EXT, '')

            f_read = open(pred_file, 'r')
            lines = f_read.readlines()
            f_read.close()

            for line in lines:
                pred_bbox = dict()
                class_name, xmin, ymin, xmax, ymax, confidence = line.rstrip().split('\t')
                pred_bbox['box'] = [float(xmin), float(ymin), float(xmax), float(ymax)]
                pred_bbox['class'] = class_name
                pred_bbox['confidence'] = float(confidence)
                pred_bbox['image_id'] = image_id
                pred_dict[class_name].append(pred_bbox)

        for class_name in pred_dict:
            pred_dict[class_name].sort(key=lambda x:float(x['confidence']), reverse=True)

        # 3. check whether bbox is matched with target_box or not. get true pos and false pos.
        ap_per_class = dict()
        sum_AP = 0.0
        for class_name in pred_dict:
            pred_bboxes_per_class = pred_dict[class_name]

            num_preds = len(pred_bboxes_per_class)
            true_pos = [0] * num_preds
            false_pos = [0] * num_preds
            for pred_bbox_idx, pred_bbox in enumerate(pred_bboxes_per_class):
                image_id = pred_bbox['image_id']

                if opt.view_image is True:
                    img = cv2.imread(path_dict[image_id])

                gt_bboxes = gt_dict[image_id]
                max_iou = -1
                max_iou_idx = -1
                for gt_bbox_idx, gt_bbox in enumerate(gt_bboxes):
                    if gt_bbox['class'] == class_name:
                        iou = _get_iou(gt_bbox['box'], pred_bbox['box'])

                        if max_iou < iou:
                            max_iou = iou
                            max_iou_idx = gt_bbox_idx

                if max_iou >= IOU_THRESHOLD:
                    if gt_bboxes[max_iou_idx]['matched'] is False:
                        true_pos[pred_bbox_idx] = 1
                        gt_bboxes[max_iou_idx]['matched'] = True

                        # if opt.view_image is True:
                        #     cv2.rectangle(img, (pred_bbox['box'][0], pred_bbox['box'][1]),
                        #                   (pred_bbox['box'][2], pred_bbox['box'][3]), (0, 255, 0))
                        #     cv2.rectangle(img, (gt_bboxes[max_iou_idx]['box'][0], gt_bboxes[max_iou_idx]['box'][1]),
                        #                   (gt_bboxes[max_iou_idx]['box'][2], gt_bboxes[max_iou_idx]['box'][3]), (0, 0, 255))
                    else:
                        false_pos[pred_bbox_idx] = 1

                        if opt.view_image is True:
                            cv2.rectangle(img, (int(pred_bbox['box'][0]), int(pred_bbox['box'][1])),
                                          (int(pred_bbox['box'][2]), int(pred_bbox['box'][3])), (0, 0, 255))
                            cv2.putText(img, pred_bbox['class'],
                                        (int(pred_bbox['box'][0]), int(pred_bbox['box'][1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255],
                                        1)

                            cv2.imshow(class_name, img)
                            cv2.waitKey(0)
                else:
                    false_pos[pred_bbox_idx] = 1

                    if opt.view_image is True:
                        cv2.rectangle(img, (int(pred_bbox['box'][0]), int(pred_bbox['box'][1])),
                                      (int(pred_bbox['box'][2]), int(pred_bbox['box'][3])), (255, 0, 255))
                        cv2.putText(img, pred_bbox['class'],
                                    (int(pred_bbox['box'][0]), int(pred_bbox['box'][1]) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255],
                                    1)

                        cv2.imshow(class_name, img)
                        cv2.waitKey(0)


            # 4. get mAP per class(voc style)
            # TODO: write out recall/precision for each class.
            cumsum = 0
            for idx, val in enumerate(false_pos):
                false_pos[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(true_pos):
                true_pos[idx] += cumsum
                cumsum += val
            recall = true_pos[:]
            for idx, val in enumerate(true_pos):
                if class_name not in gt_boxes_per_class:
                    print(class_name + ' is not exist in GT. Recall will be zero.')
                    recall[idx] = 0.0
                else:
                    recall[idx] = float(true_pos[idx]) / gt_boxes_per_class[class_name]
            precision = true_pos[:]
            for idx, val in enumerate(true_pos):
                precision[idx] = float(true_pos[idx]) / (false_pos[idx] + true_pos[idx])

            ap, mrec, mprec = voc_ap(recall[:], precision[:])
            ap_per_class[class_name] = ap
            sum_AP += ap

            total_tp = true_pos[-1] if len(true_pos) > 0 else 0
            total_fp = false_pos[-1] if len(false_pos) > 0 else 0

            if class_name not in gt_boxes_per_class:
                print('%-10s | AP: %.4f | #GT: %-4d | TP: %-4d | FP: %-4d'
                    % (class_name, ap, 0, total_tp, total_fp))
                not_exist_class += 1
            else:
                print('%-10s | AP: %.4f | #GT: %-4d | TP: %-4d | FP: %-4d'
                    % (class_name, ap, gt_boxes_per_class[class_name], total_tp, total_fp))

            if opt.view_result is True:
                plt.plot(recall, precision, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                text = 'AP for ' + str(class_name) +': {0:.4f}'.format(ap)
                plt.title(text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                plt.show()
                cv2.waitKey(0)

        mean_AP = sum_AP / (num_classes - not_exist_class)
        print('mAP: %.4f' % (mean_AP))

        # 5. store recall and precision value for plot

        # 6. store in mlflow
        if opt.mlflow is True and dataset_name == 'test':
            with mlflow.start_run() as run:
                mlflow.log_params(config['params'])
                mlflow.log_params(config['model'])
                mlflow.log_params(config['data'])
                mlflow.log_metric('mAP', mean_AP)
                mlflow.log_metrics(ap_per_class)
