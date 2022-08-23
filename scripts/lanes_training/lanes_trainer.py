import torch
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from src.models import get_model
from src.cityscape_dataset import CityscapeDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utills import stats_overall_accuracy, stats_iou_per_class
from src.utills import Logger
# from src.criterions.iou_loss import IoULoss
# from src.criterions.lovasz_loss import LovaszSoftmax
# from src.criterions.boundary_loss import SoftDiceLoss
from src.criterions.ND_Crossentropy import CrossentropyND
# from src.criterions.ND_Crossentropy import WeightedCrossEntropyLoss
import json
import os
import shutil


class LanesSegmentationTrainer(object):
    def __init__(self, cuda, train_settings, val_settings, model_settings, dataset_dir):
        self.train_settins = train_settings
        self.val_settings = val_settings
        self.model_settings = model_settings

        # set device
        device = torch.device("cuda:0" if cuda else "cpu")

        # set model
        self.model = get_model(model_settings["name"], model_settings["kwargs"]).to(device)
        # self.model = get_model(model_settings["name"], model_settings["kwargs"]).cuda()

        # set criterion
        self.criterion = CrossentropyND()

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_settins["lr"])

        # logger
        self.logger = Logger(("train", "val", "iou", "overall_accuracy"), "loss")

        # train dataloader
        train_dataset = CityscapeDataset(dataset_dir, train_settings["type"])
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.train_settins["batch_size"],
                                                            shuffle=True,
                                                            num_workers=self.train_settins["num_workers"],
                                                            drop_last=True,
                                                            pin_memory=True if cuda else False)

        # val dataloader
        val_dataset = CityscapeDataset(dataset_dir, val_settings["type"])
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.val_settings["batch_size"],
                                                          shuffle=True,
                                                          num_workers=self.val_settings["num_workers"],
                                                          drop_last=True,
                                                          pin_memory=True if cuda else False)

    def train(self, epoch):
        self.model.train()

        for param_group in self.optimizer.param_groups:
            print("learning rate: {}".format(param_group["lr"]))

        cm = np.zeros((self.model_settings["kwargs"]["out_channels"],
                       self.model_settings["kwargs"]["out_channels"]))
        t = tqdm(self.train_dataloader, ncols=100, desc=f"Epoch {epoch}")

        # loss
        loss_meter = 0

        for inputs, targets in t:
            inputs = inputs.cuda()
            targets = targets.cuda()

            targets = fn.resize(targets, self.train_settins["size"])

            outputs = self.model(inputs)
            self.optimizer.zero_grad()
            # loss = F.cross_entropy(outputs, targets)
            # loss.backward()
            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()

            outputs_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            targets_np = targets.cpu().numpy()

            cm += confusion_matrix(targets_np.ravel(),
                                   outputs_np.ravel(),
                                   labels=list(range(self.model_settings["kwargs"]["out_channels"])))

            oa = stats_overall_accuracy(cm)
            avIoU = stats_iou_per_class(cm)[0]

            t.set_postfix(OA=f"{oa:.3f}", AvIOU=f"{avIoU:.3f}")

            loss_meter += loss.float()

        # oa = stats_overall_accuracy(cm)
        # avIoU = stats_iou_per_class(cm)[0]
        loss_meter /= self.train_settins["batch_size"]

        return loss_meter.cpu().detach().numpy()  # , oa, avIoU

    def val(self, epoch):
        self.model.eval()

        cm = np.zeros((self.model_settings["kwargs"]["out_channels"],
                       self.model_settings["kwargs"]["out_channels"]))
        t = tqdm(self.val_dataloader, ncols=100, desc=f"Epoch {epoch}")

        # loss
        loss_meter = 0

        with torch.no_grad():
            for inputs, targets in t:
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets = fn.resize(targets, self.train_settins["size"])

                outputs = self.model(inputs)

                # loss = F.cross_entropy(outputs, targets)
                loss = self.criterion(outputs, targets)

                outputs_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                targets_np = targets.cpu().numpy()

                cm += confusion_matrix(targets_np.ravel(),
                                       outputs_np.ravel(),
                                       labels=list(range(self.model_settings["kwargs"]["out_channels"])))

                oa = stats_overall_accuracy(cm)
                avIoU = stats_iou_per_class(cm)[0]

                t.set_postfix(OA=f"{oa:.3f}", AvIOU=f"{avIoU:.3f}")

                loss_meter += loss.float()

            oa = stats_overall_accuracy(cm)
            avIoU = stats_iou_per_class(cm)[0]
            loss_meter /= self.train_settins["batch_size"]

        return loss_meter.cpu().detach().numpy(), oa, avIoU

    def save_checkpoint(self, state, is_best, name="checkpoint.pth"):
        print("Saving checkpoint")
        file_name = os.path.join(self.train_settins["save_dir"], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(self.train_settins["save_dir"], "best_iou_model.pth"))

    def start_training(self):
        start_epoch = 0
        best_iou = 0

        if self.train_settins["resume_path"] is not None and os.path.exists(self.train_settins["resume_path"]):
            print("Resuming from {}".format(self.train_settins["resume_path"]))
            state = torch.load(self.train_settins["resume_path"])
            start_epoch = state["epoch"] + 1
            best_iou = state["best_iou"]
            self.model.load_state_dict(state["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(state["model_state_dict"])
            self.logger.data = state["logger_data"]

        for epoch in range(start_epoch, self.train_settins["n_epochs"]):
            print("Starting epoch {}".format(epoch))
            train_loss = self.train(epoch)
            val_loss, val_oa, val_iou = self.val(epoch)

            print("===> train loss: {:.2f}".format(train_loss))
            print("===> val loss: {:.2f}, val oa: {:.2f}, val iou: {:.2f}".format(val_loss, val_oa, val_iou))

            self.logger.add("train", train_loss)
            self.logger.add("val", val_loss)
            self.logger.add("iou", val_iou)
            self.logger.add("overall_accuracy", val_oa)
            self.logger.plot(save=self.train_settins["save"], save_dir=self.train_settins["save_dir"])

            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)

            if self.train_settins["save"]:
                state = {
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "logger_data": self.logger.data
                }
                self.save_checkpoint(state, is_best)


if __name__ == "__main__":
    print("Loading configurations")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as config:
        config = json.load(config)

        lane_segmentation_trainer = LanesSegmentationTrainer(config["cuda"],
                                                             config["train_settings"],
                                                             config["val_settings"],
                                                             config["model_settings"],
                                                             config["dataset_dir"])
        lane_segmentation_trainer.start_training()
