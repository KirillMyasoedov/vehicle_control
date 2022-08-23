import os


def remove_redundant(reference_dir, removal_dir, type):
    removal_files = os.listdir(os.path.join(removal_dir, type))
    reference_files = os.listdir(os.path.join(reference_dir, type))

    for file in removal_files:
        if ".npz" in file:
            reference_file = file.replace(".npz", "")
        else:
            reference_file = file + ".npz"
        if reference_file not in reference_files:
            file_path = os.path.join(removal_dir, type, file)
            os.remove(file_path)


masks_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/rgb"
labels_dir = "/home/kirill/catkin_ws/src/vehicle_control/lanes_dataset/labels"

remove_redundant(labels_dir, masks_dir, "train")
