import torch
from torchvision import models


def create_model(outputchannels=1, aux_loss=False, freeze_backbone=False):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True, aux_loss=aux_loss)

    if freeze_backbone is True:
        for p in model.parameters():
            p.requires_grad = False

    model.classifier = models.segmentation.segmentation.DeepLabHead(
        2048, outputchannels)

    return model
