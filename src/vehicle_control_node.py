import rospy


class VehicleControlNode(object):
    def __init__(self,
                 input_data_adapter,
                 images_segmenter,
                 output_data_adapter,
                 dataset_creator,
                 create_dataset,
                 period=0.1):
        self.input_data_adapter = input_data_adapter
        self.images_segmenter = images_segmenter
        self.output_data_adapter = output_data_adapter
        self.dataset_creator = dataset_creator
        self.create_dataset = create_dataset

        print("Starting segmentation")
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_callback)

    def timer_callback(self, _):
        rgb_image = self.input_data_adapter.get_rgb_image()
        testing_mask = self.input_data_adapter.get_segmented_image()

        if rgb_image is None or testing_mask is None:
            return None

        if self.create_dataset:
            self.dataset_creator.save_images(rgb_image, testing_mask)
        else:
            resulting_mask = self.images_segmenter.segment_image(rgb_image, testing_mask)
            self.output_data_adapter.publish_segmented_image(resulting_mask)
