from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import mmseg_transforms as mmtf
import cv2
import mmcv

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Segmentation(Dataset):
    
    def __init__(self,
                 args,
                 base_dir='/root/share/SpaceNet_dataset/',
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'img_dir/train/')
        self._cat_dir = os.path.join(self._base_dir, 'ann_dir/train/')
        self._val_img_dir = os.path.join(self._base_dir, 'img_dir/val_1024/')
        self._val_cat_dir = os.path.join(self._base_dir, 'ann_dir/val_1024/')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        self._splits_dir = os.path.join(self._base_dir, 'splits/')
        self.args = args

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            
            if splt == 'train':
                lines = []
                with open(os.path.join(os.path.join(self._splits_dir, 'train.txt')), "r") as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    _image = os.path.join(self._image_dir, line + '.png')
                    _cat = os.path.join(self._cat_dir, line + '.png')
                    self.im_ids.append(line)
                    self.images.append(_image)
                    self.categories.append(_cat)
            else:
                lines = []
                with open(os.path.join(os.path.join(self._splits_dir, 'val1.txt')), "r") as f:
                    lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    _image = os.path.join(self._val_img_dir, line + '.png')
                    _cat = os.path.join(self._val_cat_dir, line + '.png')
                    self.im_ids.append(line)
                    self.images.append(_image)
                    self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size


    def __getitem__(self, index):
        for splt in self.split:
            if splt == "train":
                _img, _target = self._make_img_gt_point_pair(index)
                sample = {'img': _img, 'label': _target, 'seg_fields': ['label']}
                return self.transform_tr(sample)
            elif splt == 'val':
                _img, _target = self._make_img_gt_point_pair_val(index)
                sample = {'img': _img, 'label': _target, 'seg_fields': ['label']}
                return self.transform_val(sample)
            else:
                _img, _target = self._make_img_gt_point_pair_val(index)
                sample = {'img': _img, 'label': _target, 'seg_fields': ['label']}
                return self.transform_test(sample), self.im_ids[index]


    def _make_img_gt_point_pair(self, index):
        _img = cv2.imread(self.images[index])
        _target = cv2.imread(self.categories[index], cv2.IMREAD_GRAYSCALE)
        return _img, _target


    def _make_img_gt_point_pair_val(self, index):
        _img = cv2.imread(self.images[index])
        _target = cv2.imread(self.categories[index], cv2.IMREAD_GRAYSCALE)
        return _img, _target


    def transform_tr(self, sample):
        crop_size = (self.args.crop_size, self.args.crop_size)
        img_norm_cfg = dict(mean=[74.66660569, 75.07802649, 75.56446617], std=[46.5538831, 48.33664143, 50.01950716],
                            to_rgb=True)
        composed_transforms = transforms.Compose([
            mmtf.Resize(img_scale=(self.args.base_size, self.args.base_size), ratio_range=(0.8, 1.2)),
            mmtf.RandomCrop(crop_size=crop_size),
            mmtf.RandomFlip(prob=0.5, direction='horizontal'),
            mmtf.RandomFlip(prob=0.5, direction='vertical'),
            mmtf.PhotoMetricDistortion(),
            mmtf.Normalize(**img_norm_cfg),
            mmtf.Rerange(min_value=0.0, max_value=1.0, for_seg_field=True),
            mmtf.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_val(self, sample):
        img_norm_cfg_val = dict(
            mean=[74.66660569, 75.07802649, 75.56446617], std=[46.5538831, 48.33664143, 50.01950716], to_rgb=True)
        composed_transforms = transforms.Compose([
            mmtf.Normalize(**img_norm_cfg_val),
            mmtf.Rerange(min_value=0.0, max_value=1.0, for_seg_field=True),
            mmtf.ToTensor()
        ])

        return composed_transforms(sample)

    def transform_test(self, sample):
        img_norm_cfg_val = dict(
            mean=[74.66660569, 75.07802649, 75.56446617], std=[46.5538831, 48.33664143, 50.01950716], to_rgb=True)
        composed_transforms = transforms.Compose([
            mmtf.Normalize(**img_norm_cfg_val),
            mmtf.Rerange(min_value=0.0, max_value=1.0, for_seg_field=True),
            mmtf.ToTensor()
        ])

        return composed_transforms(sample)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'spacenet'
    args.batch_size = 4
    args.base_size = 1300
    args.crop_size = 1024

    voc_train = Segmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=4, shuffle=True, num_workers=0)
