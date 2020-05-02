import torch
from torchvision import models


def create_model(outputchannels=1, aux_loss=False):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True, aux_loss=aux_loss)

    model.classifier = models.segmentation.segmentation.DeepLabHead(
        2048, outputchannels)

    return model
