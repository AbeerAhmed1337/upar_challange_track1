import os
import pickle
import glob
import random

import numpy as np
import torch.utils.data as data
from PIL import Image

import torchvision.transforms as T


class UPAR(data.Dataset):
    """
    Load UPAR dataset from pickle file.

    split: train/val/trainval/test split
    partition: partition id 0-8 for the requested split
    root: path to datasets, original datasets must be in this directory
    data_path: path to UPAR pickle file
    transform: training data transforms
    target_transform: evaluation data transforms
    """
    def __init__(self, split='train', partition=0, root='datasets', data_path='./data/UPAR/dataset_all.pkl', transform=None, target_transform=None):
        with open(data_path, 'rb') as f:
            dataset_info = pickle.load(f)

        self.dataset_info = dataset_info

        if isinstance(dataset_info, dict):
            img_id = dataset_info['image_name']
            attr_label = dataset_info['label']
            partition_info = dataset_info['partition']
            self.attr_id = dataset_info['attr_name']
            dataset_ids = dataset_info.get('dataset_ids', np.arange(len(img_id)))
        else:
            img_id = dataset_info.image_name
            attr_label = dataset_info.label
            partition_info = dataset_info.partition
            self.attr_id = dataset_info.attr_name
            dataset_ids = getattr(dataset_info, 'dataset_ids', np.arange(len(img_id)))

        if split not in partition_info:
            raise AssertionError(f'split {split} does not exist')

        self.dataset = 'UPAR'
        self.transform = transform
        self.target_transform = target_transform
        self.root_path = root
        self.attr_num = len(self.attr_id)
        self.dataset_ids = dataset_ids

        split_partitions = partition_info[split]
        if isinstance(split_partitions, list) and len(split_partitions) > 0 and isinstance(split_partitions[0], (list, tuple, np.ndarray)):
            if partition < 0 or partition >= len(split_partitions):
                raise IndexError(f'partition {partition} out of range for split {split}')
            self.img_idx = np.asarray(split_partitions[partition], dtype=np.int64)
        else:
            self.img_idx = np.asarray(split_partitions, dtype=np.int64)

        if self.img_idx.ndim > 1:
            self.img_idx = np.hstack(self.img_idx)

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

        test_partitions = partition_info.get('test', []) if isinstance(partition_info, dict) else getattr(partition_info, 'test', [])
        if isinstance(test_partitions, list) and len(test_partitions) > 0 and partition < len(test_partitions):
            test_partition = test_partitions[partition]
            if isinstance(test_partition, list) and len(test_partition) > 0 and isinstance(test_partition[0], (list, tuple, np.ndarray)):
                self.sub_dataset_lengths = [len(d) for d in test_partition]
            else:
                self.sub_dataset_lengths = [len(test_partition)]
        else:
            self.sub_dataset_lengths = [len(self.img_idx)]

    def __getitem__(self, index):
        imgname = self.img_id[index]
        gt_label = self.label[index]
        imgidx = self.img_idx[index]
        did = self.dataset_ids[imgidx]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.target_transform(gt_label)

        return img, gt_label, did, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(height, width):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = [
        T.Resize((height, width))
    ]
	
    train_transform += [
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
    ]

    train_transform += [
        T.ToTensor(),
        normalize,
    ]
    train_transform = T.Compose(train_transform)

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
