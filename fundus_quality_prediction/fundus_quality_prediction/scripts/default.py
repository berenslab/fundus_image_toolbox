import os

# Data root. In there, folders "bad" and "good" with subfolders "drimdb" and "deepdrid-isbi2020" 
# must exist.
DATA_ROOT = os.path.join(os.path.dirname(__file__), "../data").__str__()
# Directory that holds model folders
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models").__str__()
# Model folders names for 10-ensemble
ENSEMBLE_MODELS = ['2024-05-03 14-38-34', '2024-05-03 16-23-29', '2024-05-03 14-58-37', '2024-05-03 14-25-19', '2024-05-03 15-28-00', '2024-05-03 14-25-42', '2024-05-03 15-52-56', '2024-05-03 15-04-29', '2024-05-06 20-25-14', '2024-05-03 15-59-31']
