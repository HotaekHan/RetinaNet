import json

is_out = True

datasets = [
    'coco_val2014_table.json',
    'coco_val2017_table.json'
]

out_path = 'coco_val1417_table.json'

out_dict = dict()
out_idx = 0

for dataset in datasets:
    fp_read = open(dataset, 'r')
    gt_dict = json.load(fp_read)

    for gt_key in gt_dict:
        gt_data = gt_dict[gt_key]

        new_key = 'img_' + str(out_idx)
        out_dict[new_key] = gt_data
        out_idx += 1

if is_out is True:
    json_str = json.dumps(out_dict, indent=4)
    out_file = open(out_path, 'w')
    out_file.writelines(json_str)
    out_file.close()