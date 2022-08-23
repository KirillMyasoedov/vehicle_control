import torch
from src.models import get_model
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import glob


class CityscapeTestDataset(Dataset):
    def __init__(self, root_dir="./", type="train"):
        self.type = type
        self.images_dir = os.path.join(root_dir, "rgb/{}".format(type))
        print("Creating cityscapes dataset")
        # get image and labels list
        image_list = os.listdir(self.images_dir)
        image_list.sort()
        self.image_list = image_list

    def __len__(self):
        self.size = len(self.image_list)
        return self.size

    def __getitem__(self, item):
        # load image
        image = np.array(Image.open(os.path.join(self.images_dir, self.image_list[item])))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5
        image = torch.from_numpy(image).float()

        return image, self.image_list[item]


class LanesSegmentationTester(object):
    def __init__(self, cuda, test_settings, model_settings, dataset_dir, checkpoint_dir):
        self.test_settings = test_settings
        self.model_settings = model_settings

        # set device
        device = torch.device("cuda:0" if cuda else "cpu")

        # set model
        self.model = get_model(model_settings["name"], model_settings["kwargs"]).to(device)

        if os.path.exists(os.path.join(checkpoint_dir, "best_iou_model.pth")):
            state = torch.load(os.path.join(checkpoint_dir, "best_iou_model.pth"))
            self.model.load_state_dict(state["model_state_dict"], strict=True)

        # train dataloader
        test_dataset = CityscapeTestDataset(dataset_dir, self.test_settings["type"])
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=self.test_settings["batch_size"],
                                                           shuffle=True,
                                                           num_workers=self.test_settings["num_workers"],
                                                           drop_last=True,
                                                           pin_memory=True if cuda else False)

    def test(self):
        self.model.eval()
        t = tqdm(self.test_dataloader, ncols=100)

        with torch.no_grad():
            for inputs, name in t:
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                outputs_np = np.argmax(outputs.cpu().detach().numpy(), axis=1).squeeze()

                output_mask = np.zeros((600, 800, 3), dtype=np.uint8)
                output_mask[outputs_np == 1] = [255, 0, 0]
                output_mask[outputs_np == 2] = [0, 255, 0]
                output_mask[outputs_np == 3] = [0, 0, 255]

                output_image = Image.fromarray(output_mask)

                np.savez(os.path.join(self.test_settings["save_dir"], "labels", name[0]), outputs_np)
                output_image.save(os.path.join(self.test_settings["save_dir"], "images", name[0]))


if __name__ == "__main__":
    print("Loading configurations")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as config:
        config = json.load(config)

        lane_segmentation_tester = LanesSegmentationTester(config["cuda"],
                                                           config["test_settings"],
                                                           config["model_settings"],
                                                           config["dataset_dir"],
                                                           config["train_settings"]["save_dir"])
        lane_segmentation_tester.test()
