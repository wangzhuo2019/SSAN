import numpy as np
import cv2
from torchvision import transforms
import random
import torch


class Normaliztion_valtest_video(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x = sample['image_x']
        image_x = (image_x - 127.5)/128     # [-1,1]
        sample['image_x'] = image_x
        sample['map_x'] = sample['map_x']/255.0
        return sample


class Normaliztion_valtest_video_ImageNet(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __init__(self):
        self.trans = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    def __call__(self, sample):
        image_x = sample['image_x']/255
        image_x = self.trans(image_x)
        sample['image_x'] = image_x
        return sample


class ToTensor_valtest_video(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['label']
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        val_map_x = np.array(val_map_x)
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        sample['image_x'] = torch.from_numpy(image_x.astype(np.float)).float()
        sample['map_x'] = torch.from_numpy(val_map_x.astype(np.float)).float()
        sample['label'] = torch.from_numpy(spoofing_label_np.astype(np.long)).long()
        return sample


def transformer_test_video():
    return transforms.Compose([Normaliztion_valtest_video(), ToTensor_valtest_video()])


# use ImageNet mean and var for normalization
def transformer_test_video_ImageNet():
    return transforms.Compose([ToTensor_valtest_video(), Normaliztion_valtest_video_ImageNet()])