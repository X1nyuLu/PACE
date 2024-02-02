"""
These transforms are derived and modified from SANet(https://github.com/DenglinGo/bacteria-SANet)
"""

import random
import numpy as np
import torch
from torchvision import transforms
from scipy import signal


class SpecResize:
    def __init__(self, new_length):
        self.new_length = new_length

    def __call__(self, inp):
        inp = transforms.Resize([1, self.new_length])(inp.reshape(1, 1, -1))
        return inp.reshape(1, -1)


class SpecToTensor:
    def __call__(self, inp):
        inp = torch.tensor(inp, dtype=torch.float32)
        if len(inp.shape) < 3:
            inp = inp.reshape(1, -1)
        return inp


class Smooth:
    def __call__(self, inp):
        inp = signal.savgol_filter(inp, 7, 3)
        return inp


class RandomMask:
    def __init__(self, region):
        self.region = region

    def __call__(self, inp):
        start = torch.randint(0, inp.shape[-1], (1,))
        length = torch.randint(0, self.region, (1,))
        inp[start:start + length] = 0
        return inp


# Add graussian noise with zero mean and standard deviation 0.01 to 0.04
class AddGaussianNoise(object):
    def __call__(self, x):
        var = random.random() * 0.04 + 0.01
        noise = np.random.normal(0, var, (1000))
        x += noise
        x = np.clip(x, 0, 1)
        return x


# Average blur with widow size 1 to 5
class RandomBlur(object):
    def __call__(self, x):
        size = random.randint(1, 5)
        x = np.convolve(x, np.ones(size) / size, mode='same')
        return x


# randomly set the intensity of spectrum to 0
class RandomDropout(object):
    def __call__(self, x, droprate=0.1):
        noise = np.random.random(len(x))
        x = (noise > droprate) * x
        return x


# mulitiply the spectrum by a scale-factor
class RandomScaleTransform(object):
    def __call__(self, x):
        scale = np.random.uniform(0.9, 1.1, x.shape)
        x = scale * x
        x = np.clip(x, 0, 1)
        return x


# convert to Tensor with 1 channel
class ToFloatTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).view(1, -1).float()


bacteria_train_transform = transforms.Compose([
    Smooth(),
    transforms.RandomApply([transforms.RandomChoice([
        RandomBlur(),
        AddGaussianNoise()])], p=0.5),
    transforms.RandomApply([RandomScaleTransform()], p=0.5),
    SpecToTensor(),
    transforms.RandomApply([RandomMask(8)], p=0.5)]
)

bacteria_valid_transform = transforms.Compose([Smooth(), SpecToTensor()])
