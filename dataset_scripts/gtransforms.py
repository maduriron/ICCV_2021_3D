import torchvision
import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F
import numpy as np
import PIL
from torchvision import transforms
import random


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)
        self.size = size
        
    def __call__(self, img_group):
        return [self.worker(img) if img.size[0] != self.size[0] else img for img in img_group]
    
class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images
    
class GroupContrast(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = random.uniform(0.8, 1.0)
        return [PIL.ImageEnhance.Contrast(img).enhance(v) for img in img_group]

class GroupShearX(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = np.random.uniform(-0.05, 0.05)
        if random.random() > 0.5:
            v = -v
        return [img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)) for img in img_group]
    
class GroupShearY(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = np.random.uniform(-0.05, 0.05)
        if random.random() > 0.5:
            v = -v
        return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)) for img in img_group]
    
class GroupCutOut(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = np.random.uniform(0, 0.1)
        w, h = img_group[0].size
        v = v * w
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)
        
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        
        img_group = [img.copy() for img in img_group]
        for img in img_group:
            PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img_group
    
class GroupGaussianBlur(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        sigma = random.uniform(0.1, 0.5)
        kernel_size = 3
        return [transforms.GaussianBlur(kernel_size, sigma=sigma)(img) for img in img_group]

class GroupColor(object):
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = random.uniform(0.1, 1.9)
        return [PIL.ImageEnhance.Color(img).enhance(v) for img in img_group]
    
class GroupBrightness:
    def __call__(self, img_group):
        prob = random.uniform(0, 1)
        if prob < 0.5:
            return img_group
        v = random.uniform(1.8, 1.9)
        return [PIL.ImageEnhance.Brightness(img).enhance(v) for img in img_group]
    
class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return img_group
    
class GroupRandomVerticalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
        return img_group
    
class GroupTranspose(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.TRANSPOSE) for img in img_group]
        return img_group

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor): # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor
    

class LoopPad(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.size(0)

        if length==self.max_len:
            return tensor

        # repeat the clip as many times as is necessary
        n_pad = self.max_len - length
        pad = [tensor]*(n_pad//length)
        if n_pad%length>0:
            pad += [tensor[0:n_pad%length]]

        tensor = torch.cat([tensor]+pad, 0)
        return tensor

class ToTensor(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x)*255

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)
    
    
class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated
