import os
import shutil
from PIL import Image
import numpy as np


dataset_dir = "/dataset/segmented/train"
labels_dir = "/dataset/labels/train"
lanes_dataset_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/images/train"
lanes_mask_dir = "/lanes_dataset/masks"

files = os.listdir(dataset_dir)
labeling_files = files[:30]

for file in labeling_files:
    src_file = os.path.join(dataset_dir, file)
    dst_file = os.path.join(lanes_dataset_dir, file)
    segmentation_file = os.path.join(labels_dir, file) + ".npz"
    lanes_mask_file = os.path.join(lanes_mask_dir, file)

    shutil.copyfile(src_file, dst_file)

    lane_mask = np.zeros((600, 800, 3), dtype=np.uint8)
    segmentation_map = np.load(segmentation_file)["arr_0"]
    lane_mask[segmentation_map == 6] = [225, 225, 225]
    lane_image = Image.fromarray(lane_mask)
    lane_image.save(lanes_mask_file)
