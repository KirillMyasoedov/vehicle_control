import rospy
from sensor_msgs.msg import Image
import ros_numpy


def create_image_message(image):
    image_message = ros_numpy.msgify(Image, image)

    return image_message


class OutputDataAdapter(object):
    def __init__(self,
                 segmented_images_topic="/segmented_images"):
        self.segmented_images_publisher = rospy.Publisher(segmented_images_topic, Image, queue_size=10)

    def publish_segmented_images(self, image):
        image_message = create_image_message(image)
        self.segmented_images_publisher.publish(image_message)
