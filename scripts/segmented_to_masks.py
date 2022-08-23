import os
import numpy as np
from PIL import Image


def create_masks(src_dir, dst_dir, type):
    files = os.listdir(os.path.join(src_dir, type))
    for file in files:
        src_file_path = os.path.join(src_dir, type, file)
        dst_file_path = os.path.join(dst_dir, file.replace(".npz", ""))

        mask = np.zeros((600, 800, 3), dtype=np.uint8)

        labels_array = np.load(src_file_path)["arr_0"]
        mask[labels_array == 6] = [255, 255, 255]

        mask_image = Image.fromarray(mask)
        mask_image.save(dst_file_path)


labels_dir = "/home/kirill/catkin_ws/src/vehicle_control/dataset/labels"
destination_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/rgb/test"

create_masks(labels_dir, destination_dir, "train")
create_masks(labels_dir, destination_dir, "val")
