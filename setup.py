from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

settings = generate_distutils_setup(
    packages=["src",
              "src.models",
              "src.utills"],
)

setup(install_requires=["numpy",
                        "torch",
                        "Pillow",
                        "random2",
                        "tqdm",
                        "sklearn",
                        "matplotlib",
                        "pandas",
                        "pyyaml",
                        "rospkg",
                        "carla",
                        "transforms3d",
                        "opencv-python",
                        "empy",
                        "torchvision"],
      **settings)
