import numpy as np
import cv2
from torchvision import transforms
import random
import torch


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x = sample['image_x']
        map_x = sample['map_x']
        image_x = (image_x - 127.5)/128     # [-1,1]
        sample['image_x'] = image_x
        new_map_x = map_x/255.0
        sample['map_x'] = new_map_x
        return sample


class Normaliztion_ImageNet(object):
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
        map_x = sample['map_x']
        new_map_x = map_x/255.0
        sample['map_x'] = new_map_x
        return sample

    
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['label']
        # swap color, bgr to rgb axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        map_x = np.array(map_x)
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        sample['image_x'] = torch.from_numpy(image_x.astype(np.float)).float()
        sample['map_x'] = torch.from_numpy(map_x.astype(np.float)).float()
        sample['label'] = torch.from_numpy(spoofing_label_np.astype(np.long)).long()
        return sample


class Cutout(object):

    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img = sample['image_x']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        sample['image_x'] = img
        return sample


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        if random_float(0.0, 1.0) < 0.5:
            image_x = sample["image_x"]
            map_x = sample["map_x"]
            image_x = cv2.flip(image_x, 1)
            map_x = cv2.flip(map_x, 1)
            sample["image_x"] = image_x
            sample["map_x"] = map_x
        return sample


class Contrast_and_Brightness(object):

    def __call__(self, sample):
        image_x = sample["image_x"]
        gamma = random.randint(-40, 40)
        alpha = random_float(0.5, 1.5)
        image_x = cv2.addWeighted(image_x, alpha, image_x, 0, gamma)
        sample["image_x"] = image_x
        return sample


def random_float(f_min, f_max):
    return f_min + (f_max-f_min) * random.random()


def transformer_train():
    return transforms.Compose([Contrast_and_Brightness(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()])


def transformer_train_pure():
    return transforms.Compose([RandomHorizontalFlip(), ToTensor(), Normaliztion()])


def transformer_train_ImageNet():
    return transforms.Compose([RandomHorizontalFlip(), ToTensor(), Normaliztion_ImageNet()])