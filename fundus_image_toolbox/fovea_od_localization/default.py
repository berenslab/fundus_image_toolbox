# import os
from pathlib import Path
from types import SimpleNamespace

DEFAULT_MODEL = "2024-05-07 11_13.05"
# DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "data/ADAM+IDRID+REFUGE_df.csv").__str__()
# MODELS_DIR = os.path.join(os.path.dirname(__file__), "models").__str__()
DEFAULT_CSV_PATH = Path(__file__).parent / "data/ADAM+IDRID+REFUGE_df.csv"
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_CONFIG = SimpleNamespace(
    **{
        "batch_size": 8,
        "csv_path": DEFAULT_CSV_PATH,
        # "data_root": "../../",
        "data_root": Path(__file__).parent.parent,
        "device": "cuda:0",
        "epochs": 500,
        "img_size": 350,
        "lr": 0.0001,
        "model_type": "efficientnet-b3",
        "seed": 123,
        "testset_eval": True,
    }
)
