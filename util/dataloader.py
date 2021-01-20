from natsort import natsorted
from PIL import Image
from pathlib import Path
import torch.utils.data
import matplotlib.pyplot as plt
import os
import numpy as np

class domain_dataloader():
    def __init__(self, opt):
        self.opt = opt
        self.input_root = Path(opt.input_dir)
        self.scale_dirs = [self.input_root / Path("original")]

        initialized = create_scales(self.input_root, self.scale_dirs, opt)
        if not initialized:
            source_B = natsorted(os.listdir(self.scale_dirs[0] / Path("trainB")))
            for item in source_B:
                path = self.scale_dirs[0] / Path("trainB") / Path(item)
                create_scaled_variants(path, self.scale_dirs, opt)



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
        scale_dirs.append(output_scale_dir)
    return initialized


def create_scaled_variants(path, scale_dirs, opt):
    if os.path.isfile(str(path)):
        original = Image.open(str(path)).convert("RGB")
        scale_res = opt.base_resolution
        for scale, dir in enumerate(scale_dirs):
            if scale != 0:
                im = original.resize((scale_res, scale_res), Image.ANTIALIAS)
                output_path = dir / Path("trainB") / os.path.basename(path)
                im.save(output_path)
                scale_res *= opt.scale_factor