import rospy
import message_filters
import numpy as np
from sensor_msgs.msg import Image


class InputDataAdapter(object):
    def __init__(self,
                 rgb_images_topic="/carla/ego_vehicle/rgb_front/image",
                 segmented_images_topic="/carla/ego_vehicle/semantic_segmentation_front/image"):
        rgb_images_subscriber = message_filters.Subscriber(rgb_images_topic, Image)
        segmented_images_subscriber = message_filters.Subscriber(segmented_images_topic, Image)

        time_synchronizer = message_filters.TimeSynchronizer([rgb_images_subscriber, segmented_images_subscriber], 100)
        time_synchronizer.registerCallback(self.images_callback)

        self.rgb_images = None
        self.segmented_images = None

    def images_callback(self, rgb_images, segmented_images):
        self.rgb_images = rgb_images
        self.segmented_images = segmented_images

    def get_rgb_image(self):
        if self.rgb_images is None:
            return None

        height = self.rgb_images.height
        width = self.rgb_images.width
        data = self.rgb_images.data

        rgb_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
        rgb_image = rgb_image[:, :, 2::-1]

        return rgb_image

    def get_segmented_image(self):
        if self.segmented_images is None:
            return None

        height = self.segmented_images.height
        width = self.segmented_images.width
        data = self.segmented_images.data

        segmented_image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
        segmented_image = segmented_image[:, :, 2::-1]

        return segmented_image
