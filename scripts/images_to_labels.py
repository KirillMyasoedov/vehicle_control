from PIL import Image
import os
import numpy as np


def convert_images(root, destination, type):
    images_list = os.listdir(os.path.join(root, type))
    classes = {
        0: 0,
        210: 1,
        496: 2,
        162: 3,
        300: 4,
        459: 5,
        441: 6,
        320: 7,
        511: 8,
        284: 9,
        255: 10,
        360: 11,
        440: 12
    }
    for image in images_list:
        image_np = np.array(Image.open(os.path.join(root, type, image)))
        image_np = image_np.sum(axis=2)
        labels_image = np.zeros((image_np.shape[0], image_np.shape[1]))
        for key, value in classes.items():
            labels_image[np.where(image_np == key)] = value

        np.savez(os.path.join(destination, type, image), labels_image)


segmented_dir = "/dataset/segmented"
destination = "/home/kirill/catkin_ws/src/vehicle_control/dataset/labels"
convert_images(segmented_dir, destination, "train")
convert_images(segmented_dir, destination, "val")
