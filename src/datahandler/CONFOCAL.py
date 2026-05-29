import os

import cv2
import numpy as np
import torch

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class Confocal(DenoiseDataSet):
    '''
    Confocal dataset class for uint16 single-channel tif images.
    All images are stored in a single folder.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = os.path.join(self.dataset_dir, 'confocal')
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # Scan all TIF files in the dataset directory
        for file_name in sorted(os.listdir(dataset_path)):
            if file_name.lower().endswith(('.tif', '.tiff')):
                self.img_paths.append(os.path.join(dataset_path, file_name))

        assert len(self.img_paths) > 0, 'No TIF images found in %s'%dataset_path
        print('Found %d confocal images in %s' % (len(self.img_paths), dataset_path))

    def _load_data(self, data_idx):
        img_path = self.img_paths[data_idx]

        # IMREAD_UNCHANGED preserves original bit depth (16-bit for confocal)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, "failure on loading image - %s"%img_path

        # uint16: value range [0, 65535]
        img = np.expand_dims(img.astype(np.float32), axis=0)
        noisy_img = torch.from_numpy(np.ascontiguousarray(img))

        return {'real_noisy': noisy_img}


@regist_dataset
class prep_confocal(DenoiseDataSet):
    '''
    Dataset class for prepared confocal dataset which is cropped with overlap.
    [using size 512x512 with 0 overlapping]
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/confocal_s512_o0')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        img_path = os.path.join(self.dataset_path, 'RN', file_name)
        # IMREAD_UNCHANGED preserves 16-bit depth
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        assert img is not None, "failure on loading image - %s"%img_path

        img = np.expand_dims(img.astype(np.float32), axis=0)
        noisy_img = torch.from_numpy(np.ascontiguousarray(img))

        return {'real_noisy': noisy_img}
