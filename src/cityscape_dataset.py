import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import random
import numpy as np


class CityscapeDataset(Dataset):
    def __init__(self, root_dir="./", type="train"):
        self.type = type
        print("Creating cityscapes dataset")
        # get image and labels list
        image_list = glob.glob(os.path.join(root_dir, "rgb/{}".format(type), "*.png"))
        image_list.sort()
        self.image_list = image_list

        labels_list = glob.glob(os.path.join(root_dir, "labels/{}".format(type), "*.npz"))
        labels_list.sort()
        self.labels_list = labels_list

    def __len__(self):
        self.size = len(self.image_list)
        return self.size

    def __getitem__(self, item):
        # load image
        image = np.array(Image.open(self.image_list[item]))

        # load instance
        labels = np.load(self.labels_list[item])["arr_0"]

        # data augmentation
        if self.type == "train":
            if random.randint(0, 1):
                image = image[:, ::-1]
                labels = labels[:, ::-1]

        image = image.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5
        labels = labels.astype(np.int64)

        image = torch.from_numpy(image).float()
        labels = torch.from_numpy(labels).long()

        return image, labels
