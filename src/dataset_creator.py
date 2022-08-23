from PIL import Image
import os
import random
import shutil


class DatasetCreator(object):
    def __init__(self, size, directory, train_type, val_type):
        self.size = size
        self.directory = directory
        self.train_type = train_type
        self.val_type = val_type
        self.images_counter = 0

        # creating dataset directory
        if not os.path.exists(self.directory):
            os.makedirs(os.path.join(self.directory, "rgb", "train"))
            os.makedirs(os.path.join(self.directory, "rgb", "val"))
            os.makedirs(os.path.join(self.directory, "segmented", "train"))
            os.makedirs(os.path.join(self.directory, "segmented", "val"))
            os.makedirs(os.path.join(self.directory, "labels", "train"))
            os.makedirs(os.path.join(self.directory, "labels", "val"))

    def save_images(self, rgb_image, labels_image):
        if self.images_counter <= self.size:
            self.images_counter += 1

            rgb_image = Image.fromarray(rgb_image)
            labels_image = Image.fromarray(labels_image)

            rgb_image.save(os.path.join(self.directory,
                                        "rgb/{}/{}.png".format(self.train_type, self.images_counter)))
            labels_image.save(os.path.join(self.directory,
                                           "segmented/{}/{}.png".format(self.train_type, self.images_counter)))
        elif self.images_counter == self.size + 1:
            self.images_counter += 1
            rgb_saved = os.listdir(os.path.join(self.directory, "rgb/train"))
            labels_saved = os.listdir(os.path.join(self.directory, "segmented/train"))

            images = list(zip(rgb_saved, labels_saved))
            random.shuffle(images)
            rgb_saved, labels_saved = zip(*images)

            val_rgb = rgb_saved[int(-self.size / 5):]
            val_labels = rgb_saved[int(-self.size / 5):]

            for rgb, labels in zip(val_rgb, val_labels):
                shutil.copyfile(os.path.join(self.directory, "rgb/train", rgb),
                                os.path.join(self.directory, "rgb/val", rgb))
                os.remove(os.path.join(self.directory, "rgb/train", rgb))
                shutil.copyfile(os.path.join(self.directory, "segmented/train", labels),
                                os.path.join(self.directory, "segmented/val", labels))
                os.remove(os.path.join(self.directory, "segmented/train", labels))
        else:
            print("Dataset size exceeded")
