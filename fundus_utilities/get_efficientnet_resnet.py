import torch
from torchvision import models

def get_efficientnet_or_resnet(type:str, n_outs:int=1):
    """Get a torchvision EfficientNet or ResNet model initialized with ImageNet weights. 
        Choices: 
        resnet18, resnet34, resnet50, resnet101, resnet152,
        efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, 
        efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7

    Args:
        type (str): model type
        n_outs (int): number of outputs

    Returns:
        torch.nn.Module: model
    """
    if type == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(512, n_outs)
    elif type == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(512, n_outs)
    elif type == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = torch.nn.Linear(2048, n_outs)
    elif type == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V2")
        model.fc = torch.nn.Linear(2048, n_outs)
    elif type == "resnet152":
        model = models.resnet152(weights="IMAGENET1K_V2")
        model.fc = torch.nn.Linear(2048, n_outs)
    elif type == "efficientnet-b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(1280, n_outs)
    elif type == "efficientnet-b1":
        model = models.efficientnet_b1(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(1280, n_outs)
    elif type == "efficientnet-b2":
        model = models.efficientnet_b2(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(1408, n_outs)
    elif type == "efficientnet-b3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(1536, n_outs)
    elif type == "efficientnet-b4":
        model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(1792, n_outs)
    elif type == "efficientnet-b5":
        model = models.efficientnet_b5(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(2048, n_outs)
    elif type == "efficientnet-b6":
        model = models.efficientnet_b6(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(2304, n_outs)
    elif type == "efficientnet-b7":
        model = models.efficientnet_b7(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.Linear(2560, n_outs)        

    else:
        raise ValueError("Model type not supported")
    
    return model