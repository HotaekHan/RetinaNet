from pycocotools.coco import COCO
from tqdm import tqdm
import os
import yaml
import torch

from pathlib import Path
from zipfile import ZipFile
from itertools import repeat
from multiprocessing.pool import ThreadPool


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(url, f, progress=True)  # torch download
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory

    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

# Make Directories
dir = Path('/data/data/Object365')  # dataset root dir
for p in 'images', 'labels':
  (dir / p).mkdir(parents=True, exist_ok=True)
  for q in 'train', 'val':
      (dir / p / q).mkdir(parents=True, exist_ok=True)
# Download
url = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"
download([url + 'zhiyuan_objv2_train.tar.gz'], dir=dir, delete=False)  # annotations json
download([url + f for f in [f'patch{i}.tar.gz' for i in range(51)]], dir=dir / 'images' / 'train',
       curl=True, delete=False, threads=8)
# Move
train = dir / 'images' / 'train'
for f in tqdm(train.rglob('*.jpg'), desc=f'Moving images'):
  f.rename(train / f.name)  # move to /images/train
# Labels
coco = COCO(dir / 'zhiyuan_objv2_train.json')
names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
for cid, cat in enumerate(names):
  catIds = coco.getCatIds(catNms=[cat])
  imgIds = coco.getImgIds(catIds=catIds)
  for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
      width, height = im["width"], im["height"]
      path = Path(im["file_name"])  # image filename
      try:
          with open(dir / 'labels' / 'train' / path.with_suffix('.txt').name, 'a') as file:
              annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
              for a in coco.loadAnns(annIds):
                  x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                  x, y = x + w / 2, y + h / 2  # xy to center
                  file.write(f"{cid} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")
      except Exception as e:
          print(e)