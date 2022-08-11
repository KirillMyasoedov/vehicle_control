#!/usr/bin/env python3
import rospy
from src import VehicleControlNodeTrainer
from src import VehicleControlNodeFactory
import json
import os


if __name__ == '__main__':
    try:
        print("Loading configurations")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "config.json")) as config:
            config = json.load(config)

            if config["training"]:
                print("Starting training")
                vehicle_control_trainer = VehicleControlNodeTrainer(config["cuda"],
                                                                    config["train_settings"],
                                                                    config["val_settings"],
                                                                    config["model_settings"],
                                                                    config["dataset_dir"])
                vehicle_control_trainer.start_training()
            elif config["testing"] or config["creating_dataset"]:
                rospy.init_node("vehicle_control_node")
                factory = VehicleControlNodeFactory(config)
                factory.make_vehicle_control_node()
                rospy.spin()

    except rospy.ROSInterruptException:
        pass
