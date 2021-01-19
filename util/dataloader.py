from skimage import io
from natsort import natsorted
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
import torch.utils.data
import util.resize_right as resizer
import matplotlib.pyplot as plt
import os
import numpy as np

class domain_dataloader():
    def __init__(self, opt):
        self.opt = opt
        self.input_root = Path(opt.input_dir)
        self.scale_dirs = [self.input_root / Path("original")]

        for scale_num in range(1, opt.num_scales+1):
            output_scale_dir = self.input_root / Path("scale-{}".format(scale_num))
            output_scale_dir.mkdir(parents=True, exist_ok=True)
            self.scale_dirs.append(output_scale_dir)
            source_B = natsorted(os.listdir(self.scale_dirs[0] / Path("trainB")))
            for item in source_B:
                path = self.scale_dirs[0] / Path("trainB") / Path(item)
                if os.path.isfile(str(path)):
                    image = Image.open(str(path)).convert("RGB")
                    im = image.resize((128, 128), Image.ANTIALIAS)
                    im.save("test"+item)


