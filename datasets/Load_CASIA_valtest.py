import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os 
from glob import glob


frames_total = 8


def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    return region


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, transform=None, face_scale=1.3, img_size=256, map_size=32, UUID=-1):
        self.face_scale = face_scale
        self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.root_dir = root_dir
        self.map_root_dir = os.path.join(root_dir, "depth")
        self.transform = transform
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        video_name = str(self.landmarks_frame.iloc[idx, 1])
        image_dir = os.path.join(self.root_dir, video_name)
        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
        image_x, map_x = self.get_single_image_x(image_dir, video_name, spoofing_label)
        sample = {'image_x': image_x, 'label': spoofing_label, "map_x": map_x, "UUID": self.UUID}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_dir, video_name, spoofing_label):
        files_total = len([name for name in glob(os.path.join(image_dir, "*.jpg")) if os.path.isfile(os.path.join(image_dir, name))])
        interval = files_total//10
        map_dir = os.path.join(self.map_root_dir, video_name)
        image_x = np.zeros((frames_total, self.img_size, self.img_size, 3))
        map_x = np.ones((frames_total, self.map_size, self.map_size))
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            image_hair = video_name.split("/")[-1]
            for temp in range(500):
                image_name = "{}_{}.jpg".format(image_hair, image_id)
                image_path = os.path.join(image_dir, image_name)
                map_name = "{}_{}_depth.jpg".format(image_hair, image_id)
                map_path = os.path.join(map_dir, map_name)
                bbx_path = image_path.replace("jpg", "dat")
                if os.path.exists(image_path) and os.path.exists(bbx_path) and os.path.exists(map_path):
                    image_x_temp = cv2.imread(image_path)
                    map_x_temp = cv2.imread(map_path, 0)
                    if os.path.exists(image_path) and (image_x_temp is not None) and (map_x_temp is not None):
                        break
                image_id+=1
            image_x[ii,:,:,:] = cv2.resize(crop_face_from_scene(image_x_temp, bbx_path, self.face_scale), (self.img_size, self.img_size))
            temp = cv2.resize(crop_face_from_scene(map_x_temp, bbx_path, self.face_scale), (self.map_size, self.map_size))
            map_x[ii,:,:] = temp
        return image_x, map_x