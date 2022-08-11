import torch
from src.models import get_model
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utills import stats_overall_accuracy, stats_iou_per_class


class ImagesSegmenter(object):
    def __init__(self, cuda, model_settings, checkpoint_dir):
        self.model_settings = model_settings

        # set device
        device = torch.device("cuda:0" if cuda else "cpu")

        # load model
        self.model = get_model(model_settings["name"], model_settings["kwargs"]).to(device)

        # load snapshot
        if os.path.exists(os.path.join(checkpoint_dir, "best_iou_model.pth")):
            state = torch.load(os.path.join(checkpoint_dir, "best_iou_model.pth"))
            self.model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            assert False, "checkpoint_path {} does not exist!".format(checkpoint_dir)

        self.model.eval()

    def segment_image(self, rgb_image, testing_mask):
        with torch.no_grad():
            image = torch.from_numpy(rgb_image.astype(np.float32) / 255 - 0.5).float()

            # make predictions
            output = self.model(image)

            output_np = np.argmax(output.cpu().detach().numpy(), axis=1)
            cm = confusion_matrix(output_np,
                                  testing_mask,
                                  labels=list(range(self.model_settings["out_channels"])))
            oa = stats_overall_accuracy(cm)
            iou = stats_iou_per_class(cm)[0]

            print("overall accuracy: {}, iou: {}".format(oa, iou))

            return output_np
