# From https://github.com/ruc-aimc-lab/SuperRetina
"""@inproceedings{liu2022SuperRetina,
  title={Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching},
  author={Jiazhen Liu and Xirong Li and Qijie Wei and Jie Xu and Dayong Ding},
  booktitle={Proceedings of the 17th European Conference on Computer Vision (ECCV)},
  year={2022}
}""" 
from .inference import register, enhance, DEFAULT_CONFIG, WEIGHT_PATH, SuperRetina
from .inference import get_config as get_registration_config
from .inference import load_model as load_registration_model