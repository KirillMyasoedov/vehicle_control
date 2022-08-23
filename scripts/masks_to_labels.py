import os
from PIL import Image
import numpy as np
import random
import shutil


labeled_masks_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/labeled_images"
labels_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/labels"
masks_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/rgb"


def create_labels(height=600, width=800):
    labeled_masks_files = os.listdir(labeled_masks_dir)

    for labeled_mask_file in labeled_masks_files:
        labeled_mask_array = np.array(Image.open(os.path.join(labeled_masks_dir, labeled_mask_file))).astype(np.int64)
        labeled_mask_encoded = labeled_mask_array[:, :, 0] + \
                               labeled_mask_array[:, :, 1] * 255 + \
                               labeled_mask_array[:, :, 2] * 255 * 255

        labels = np.zeros((height, width))
        labels[labeled_mask_encoded == 255] = 1
        labels[labeled_mask_encoded == 255 ** 2] = 2
        labels[labeled_mask_encoded == 255 ** 3] = 3

        np.savez(os.path.join(labels_dir, "train", labeled_mask_file), labels)


def augment_data(type, counter, angle=None, res_ratio=(1, 1)):
    masks_files = sorted(os.listdir(os.path.join(masks_dir, "train")))
    labels_files = sorted(os.listdir(os.path.join(labels_dir, "train")))

    for mask_file, label_file in zip(masks_files, labels_files):
        mask_file_path = os.path.join(masks_dir, "train", mask_file)
        label_file_path = os.path.join(labels_dir, "train", label_file)

        mask_image = Image.open(mask_file_path)
        label_array = np.load(label_file_path)["arr_0"].astype(np.int8)

        if type == "v_flip":
            mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label_array = np.flip(label_array, axis=1)
            counter += 1

        elif type == "rotate":
            mask_image = mask_image.rotate(angle)
            label_image = Image.fromarray(label_array)
            label_image = label_image.rotate(angle)
            label_array = np.array(label_image)
            counter += 1
        elif type == "stretch":
            mask_image.resize((int(mask_image.size[0] * res_ratio[0]), int(mask_image.size[1] * res_ratio[1])))
            mask_image.crop((mask_image.size[0] * (res_ratio[0] - 1) / 2,
                             mask_image.size[1] * (res_ratio[1] - 1) / 2,
                             mask_image.size[0] * (3 - res_ratio[0]) / 2,
                             mask_image.size[1] * (3 - res_ratio[1]) / 2))
            label_image = Image.fromarray(label_array)
            label_image.resize((int(label_image.size[0] * res_ratio[0]), int(label_image.size[1] * res_ratio[1])))
            label_image.crop((label_image.size[0] * (res_ratio[0] - 1) / 2,
                              label_image.size[1] * (res_ratio[1] - 1) / 2,
                              label_image.size[0] * (3 - res_ratio[0]) / 2,
                              label_image.size[1] * (3 - res_ratio[1]) / 2))
            counter += 1
        else:
            assert False, "Unknown type"

        mask_image.save(os.path.join(masks_dir, "train", str(counter) + ".png"))
        np.savez(os.path.join(labels_dir, "train", str(counter) + ".png"), label_array)

    return counter


def train_val_split():
    masks_src_dir = os.path.join(masks_dir, "train")
    labels_src_dir = os.path.join(labels_dir, "train")

    masks_dst_dir = os.path.join(masks_dir, "val")
    labels_dst_dir = os.path.join(labels_dir, "val")

    mask_files = sorted(os.listdir(masks_src_dir))
    label_files = sorted(os.listdir(labels_src_dir))

    files = list(zip(mask_files, label_files))
    random.shuffle(files)

    mask_files, label_files = zip(*files)
    val_mask = mask_files[int(-len(mask_files) / 4):]
    val_label = label_files[int(-len(mask_files) / 4):]

    for mask, label in zip(val_mask, val_label):
        shutil.copyfile(os.path.join(masks_src_dir, mask), os.path.join(masks_dst_dir, mask))
        shutil.copyfile(os.path.join(labels_src_dir, label), os.path.join(labels_dst_dir, label))

        os.remove(os.path.join(masks_src_dir, mask))
        os.remove(os.path.join(labels_src_dir, label))


create_labels()

counter = 2872
counter = augment_data(type="v_flip", counter=counter)
counter = augment_data(type="rotate", counter=counter, angle=15)
counter = augment_data(type="rotate", counter=counter, angle=-15)
counter = augment_data(type="stretch", counter=counter, res_ratio=(1.5, 1))
counter = augment_data(type="stretch", counter=counter, res_ratio=(1, 1.5))

train_val_split()
