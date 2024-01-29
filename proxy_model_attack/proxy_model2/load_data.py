# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os

to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, test=False):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.test = test

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = np.float32(raw_image) / (4.0 * 255.0)
        raw_image = np.expand_dims(raw_image, axis=2)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        dslr_image = np.asarray(dslr_image)
        dslr_image = np.float32(dslr_image) / 255.0
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))

        return raw_image, dslr_image


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size):

        self.raw_dir = os.path.join(data_dir, 'test', 'huawei_full_resolution')

        self.dataset_size = size
        self.test_images = os.listdir(self.raw_dir)

        self.image_height = 1440  ##1440
        self.image_width = 1984   ##1984

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = np.float32(raw_image) / (4.0 * 255.0)
        raw_image = np.expand_dims(raw_image, axis=2)
        raw_image = raw_image[0:self.image_height, 0:self.image_width, :]

        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image

class LoadTargetData(Dataset):

    def __init__(self, dataset_dir, dataset_size):

        self.target_dir = os.path.join(dataset_dir, 'test', 'target')

        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        target_image = imageio.imread(os.path.join(self.target_dir, str(idx) + ".jpeg"))
        target_image = np.asarray(target_image)
        target_image = np.float32(target_image) / 255.0
        target_image = torch.from_numpy(target_image.transpose((2, 0, 1)))

        return target_image

class LoadSourceData(Dataset):

    def __init__(self, dataset_dir, dataset_size):

        self.source_dir = os.path.join(dataset_dir, 'test', 'source')

        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        source_image = imageio.imread(os.path.join(self.source_dir, str(idx) + ".png"))
        source_image = np.asarray(source_image)
        source_image = np.float32(source_image) / 255.0
        source_image = torch.from_numpy(source_image.transpose((2, 0, 1)))

        return source_image
