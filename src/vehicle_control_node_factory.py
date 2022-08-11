from .input_data_adapter import InputDataAdapter
from .image_segmenter import ImagesSegmenter
from .output_data_adapter import OutputDataAdapter
from .vehicle_control_node import VehicleControlNode
from .dataset_creator import DatasetCreator


class VehicleControlNodeFactory(object):
    def __init__(self, config):
        self.config = config

    def make_vehicle_control_node(self):
        input_data_adapter = InputDataAdapter()
        if self.config["testing"]:
            image_segmenter = ImagesSegmenter(self.config["cuda"],
                                              self.config["model_settings"],
                                              self.config["train_settings"]["save_dir"])
            output_data_adapter = OutputDataAdapter()
        else:
            image_segmenter = None
            output_data_adapter = None
        dataset_creator = DatasetCreator(self.config["dataset_size"],
                                         self.config["dataset_dir"],
                                         self.config["train_settings"]["type"],
                                         self.config["val_settings"]["type"])

        vehicle_control_node = VehicleControlNode(input_data_adapter,
                                                  image_segmenter,
                                                  output_data_adapter,
                                                  dataset_creator,
                                                  self.config["creating_dataset"])
