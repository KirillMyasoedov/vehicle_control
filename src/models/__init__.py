from src.models.unet import unet


def get_model(name, model_opts):
    if name == "Unet":
        model = unet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))
