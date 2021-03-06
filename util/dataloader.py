# Modified dataloader adapted from CUT-GAN's dataloading
from natsort import natsorted
from PIL import Image
from pathlib import Path
from util.data_transforms import get_transform, get_noise
import torch.utils.data
import matplotlib.pyplot as plt
import os
import random
import numpy as np


class multiscale_dataloader():
    def __init__(self, opt):
        self.opt = opt
        self.input_root = Path(opt.input_dir)
        self.scale_dirs = [self.input_root / Path("original")]
        self.scale_curr = 1

        initialized = create_scales(self.input_root, self.scale_dirs, opt)
        if not initialized:
            source_B = natsorted(os.listdir(self.scale_dirs[0] / Path("trainB")))
            for item in source_B:
                path = self.scale_dirs[0] / Path("trainB") / Path(item)
                create_scaled_variants(path, "B", self.scale_dirs, opt)
            source_A = natsorted(os.listdir(self.scale_dirs[0] / Path("trainA")))
            for item in source_A:
                path = self.scale_dirs[0] / Path("trainA") / Path(item)
                create_scaled_variants(path, "A", self.scale_dirs[:2], opt)
                create_scaled_variants(path, "C", self.scale_dirs, opt)

        self.scale_A = get_scale_images("A", self.scale_dirs, self.scale_curr)
        self.scale_B = get_scale_images("B", self.scale_dirs, self.scale_curr)
        self.A_size = len(self.scale_A)
        self.B_size = len(self.scale_B)

    def __getitem__(self, index):
        # Get image paths and load in image matrix
        A_path = self.scale_A[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.scale_B[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # Calculate random crop positions: 1 shared by A and C, and 1 for B
        crop_size = self.opt.base_resolution * (self.opt.scale_factor ** (self.scale_curr-1))
        load_size = int(crop_size * self.opt.load_multiplier)
        random_posA = [np.random.randint(load_size - crop_size + 1), np.random.randint(load_size - crop_size + 1)]
        random_posB = [np.random.randint(load_size - crop_size + 1), np.random.randint(load_size - crop_size + 1)]
        # Grab transform with parameters
        transformA = get_transform(self.opt, params={'flip': 1, 'crop_pos':random_posA,'crop_size': crop_size, 'load_size': load_size})
        transformB = get_transform(self.opt, params={'flip': 1, 'crop_pos':random_posB, 'crop_size': crop_size, 'load_size': load_size})
        noiseA = get_noise(self.opt)
        A = transformA(A_img)
        A = noiseA(A)
        B = transformB(B_img)
        dirs = A_path.split(os.sep)
        dirs[-2] = "trainC"
        C_path = os.path.join(*dirs)
        C_img = Image.open(C_path).convert('RGB')
        C = transformA(C_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'orig': C}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def increment_scale(self):
        self.scale_curr += 1
        self.scale_A = get_scale_images("A", self.scale_dirs, self.scale_curr)
        self.scale_B = get_scale_images("B", self.scale_dirs, self.scale_curr)

    def get_current_scale(self):
        return self.scale_curr


def create_scales(input_root, scale_dirs, opt):
    initialized = True
    for scale_num in range(1, opt.num_scales + 1):
        output_scale_dir = input_root / Path("scale-{}".format(scale_num))
        if not os.path.isdir(output_scale_dir):
            initialized = False
            output_scale_dir.mkdir(parents=True, exist_ok=True)
            dir_A = output_scale_dir / Path("trainA")
            dir_A.mkdir(parents=True, exist_ok=True)
            dir_B = output_scale_dir / Path("trainB")
            dir_B.mkdir(parents=True, exist_ok=True)
            dir_orig = output_scale_dir / Path("trainC")
            dir_orig.mkdir(parents=True, exist_ok=True)
        scale_dirs.append(output_scale_dir)
    return initialized


def create_scaled_variants(path, domain, scale_dirs, opt):
    if os.path.isfile(str(path)):
        original = Image.open(str(path)).convert("RGB")
        scale_res = opt.base_resolution
        for scale, dir in enumerate(scale_dirs):
            if scale != 0:
                im = original.resize((scale_res, scale_res), Image.ANTIALIAS)
                output_path = dir / Path("train{}".format(domain)) / os.path.basename(path)
                im.save(output_path)
                scale_res *= opt.scale_factor



def get_scale_images(domain, scale_dirs, scale):
    domain_path = "train{}".format(domain)
    data_dir = scale_dirs[scale] / Path(domain_path)
    instances = []
    for root, _, fnames in sorted(os.walk(data_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                instances.append(path)
    return instances

