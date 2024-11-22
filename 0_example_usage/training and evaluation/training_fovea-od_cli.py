from types import SimpleNamespace
import yaml
from fundus_image_toolbox.fovea_od_localization import (
    ODFoveaLoader,
    ODFoveaModel,
    Parser,
)

if __name__ == "__main__":
    parser = Parser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="If passed, other arguments are ignored and the model is trained using the config file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="efficientnet-b3",
        help="Type of model to train",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ],
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to train on"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, default=350, help="Size of the input image"
    )
    parser.add_argument(
        "--testset_eval",
        type=bool,
        default=True,
        help="Evaluate on test set after training",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../ADAM+IDRID+REFUGE_df.csv",
        help="Path to the csv file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../",
        help="Root folder that the csv entries refer to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for reproducibility. A different fixed seed is used for train-val-test split.",
    )
    config = parser.parse_args()

    if config.config is not None:
        config = yaml.safe_load(open(config.config, "r"))
        config = SimpleNamespace(**config)
    else:
        config = SimpleNamespace(**vars(config))

    print(f"Training {config.model_type} on {config.device}")
    if config.testset_eval:
        print("Evaluating on test set after training")

    train_dataloader, val_dataloader, test_dataloader = ODFoveaLoader(
        config
    ).get_dataloaders()
    model = ODFoveaModel(config)
    model.train(train_dataloader, val_dataloader)

    if config.testset_eval:
        model.evaluate(test_dataloader)

    model.plot_dist()
    model.plot_loss()
    model.plot_iou()
