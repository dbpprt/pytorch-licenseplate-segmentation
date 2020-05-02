import numbers
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomPerspective(object):
    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC, fill=0):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.fill = fill

    def __call__(self, image, target):
        if random.random() < self.p:
            width, height = image.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            
            image = F.perspective(image, startpoints, endpoints, self.interpolation, self.fill)
            target = F.perspective(target, startpoints, endpoints, self.interpolation, self.fill)

            return image, target
        return image, target
    
    @staticmethod
    def get_params(width, height, distortion_scale):
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        transform = T.RandomGrayscale(p=self.p)
        return transform(image), target

class RandomColorJitter(object):
    def __init__(self, p=0.25, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        if random.random() < self.p:
            transform = T.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
            image = transform(image)
        return image, target

class RandomGaussianSmoothing(object):
    def __init__(self, radius, p=0.2):
        self.p = p
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image, target):
        if random.random() < self.p:
            radius = np.random.uniform(self.min_radius, self.max_radius)
            return image.filter(ImageFilter.GaussianBlur(radius)), target
        return image, target

class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill
        self.degrees = (-degrees, degrees)

    def __call__(self, image, target):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        image = F.rotate(image, angle, self.resample, self.expand, self.center, self.fill)
        target = F.rotate(target, angle, self.resample, self.expand, self.center, self.fill)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size, fill):
        self.size = size
        self.fill = fill

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=self.fill)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
