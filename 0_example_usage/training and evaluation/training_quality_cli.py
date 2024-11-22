import os
import yaml
from pprint import pprint
from types import SimpleNamespace
from fundus_image_toolbox.quality_prediction import (
    FundusQualityModel,
    FundusQualityLoader,
    MODELS_DIR,
)

from argparse import ArgumentParser
import sys


class Parser(ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


if __name__ == "__main__":
    parser = Parser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train", type=bool, default=True)

    # Ignores these if config is passed
    parser.add_argument("--model_type", type=str, default="efficientnet-b4")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--img_size", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--balance_datasets", type=bool, default=True
    )  # balances the ocurrence of samples from the different dataset origins without changing the total number of training samples
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--use_datasets", nargs="+", default=["drimdb", "deepdrid-isbi2020"]
    )  # "all", 'areds', 'registration', 'drimdb', 'deepdrid-isbi2020'
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.config is not None:
        print("Loading config, ignoring other arguments...")
        with open(args.config) as c:
            config = yaml.safe_load(c)
            pprint(config)
    else:
        config = {}
        config["model_type"] = args.model_type
        config["epochs"] = args.epochs
        config["img_size"] = args.img_size
        config["lr"] = args.lr
        config["weight_decay"] = args.weight_decay
        config["device"] = args.device
        config["seed"] = args.seed
        config["balance_datasets"] = args.balance_datasets
        config["use_datasets"] = args.use_datasets
        config["batch_size"] = args.batch_size

    config = SimpleNamespace(**config)

    TRAIN = args.train

    model = FundusQualityModel(config)

    train_dataloader, val_dataloader, test_dataloader = FundusQualityLoader(
        config
    ).get_dataloaders()

    if TRAIN:
        model.train(train_dataloader, val_dataloader)

    else:
        print("Looking for latest model...")
        # dirs = os.listdir(MODELS_DIR)
        dirs = MODELS_DIR.iterdir()
        ckpt = sorted(dirs)[-1]
        model.load_checkpoint(ckpt)

    result = model.evaluate(test_dataloader=test_dataloader)
    pprint(result)

    model.save_summary()

    # model.plot_grid(test_dataloader)
