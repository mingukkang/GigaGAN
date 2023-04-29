# GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
# See license file or visit https://github.com/mingukkang/GigaGAN for details

# data_util.py

import os
import re
import io
import random

from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision.transforms as transforms
import glob



resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """ 
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class EvalDataset(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 crop_long_edge=False,
                 resize_size=None,
                 resizer="lanczos",
                 normalize=True,
                 load_txt_from_file=False,
                 ):
        super(EvalDataset, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.data_type = data_type
        self.resize_size = resize_size
        self.normalize = normalize
        self.load_txt_from_file = load_txt_from_file

        self.trsf_list = [CenterCropLongEdge()]
        if isinstance(self.resize_size, int):
            self.trsf_list += [transforms.Resize(self.resize_size,
                                                 interpolation=resizer_collection[resizer])]
        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5],
                                                    [0.5, 0.5, 0.5])]
        else:
            self.trsf_list += [transforms.PILToTensor()]
        self.trsf = transforms.Compose(self.trsf_list)

        self.load_dataset()

    def natural_sort(self, l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def load_dataset(self):
        if self.data_name == "coco2014":
            if self.load_txt_from_file:
                self.imagelist = self.natural_sort(glob.glob(os.path.join(self.data_dir, self.data_type, "*.%s" % "png")))
                captionfile = os.path.join(self.data_dir, "captions.txt")
                with io.open(captionfile, 'r', encoding="utf-8") as f:
                    self.captions = f.read().splitlines()
                self.data = list(zip(self.imagelist, self.captions))
            else:
                self.data = CocoCaptions(root=os.path.join(self.data_dir,
                                                        "val2014"),
                                        annFile=os.path.join(self.data_dir,
                                                            "annotations",
                                                            "captions_val2014.json"))
        else:
            root = os.path.join(self.data_dir, self.data_type)
            self.data = ImageFolder(root=root)

    def __len__(self):
        num_dataset = len(self.data)
        return num_dataset

    def __getitem__(self, index):
        if self.data_name == "coco2014":
            img, txt = self.data[index]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(txt, list):
                txt = txt[random.randint(0, 4)]
            return self.trsf(img), txt
        else:
            img, label = self.data[index]
            return self.trsf(img), int(label)