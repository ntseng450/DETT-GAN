# util functions from CUT-GAN
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os
import matplotlib.pyplot as plt

def save_scaled(opt, fakeB, datapath, scale):
    scaling = opt.scale_factor ** scale
    target_res = opt.base_resolution * scaling
    print(target_res)
    image_B = Image.fromarray(tensor2im(fakeB))
    scaled_B = image_B.resize((target_res, target_res), Image.ANTIALIAS)
    filename = os.path.basename(datapath[0])
    save_path = opt.input_dir / Path("scale-{}".format(scale+1)) / Path("trainA") / filename
    scaled_B.save(save_path)


def save_snapshot_visual(visuals, epoch, i):
    fig = plt.figure()
    ncols = 4
    if ncols > 0:
        ncols = min(ncols, len(visuals))
        idx = 0
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            subplot_fig = fig.add_subplot(1, ncols, idx+1)
            subplot_fig.set_title(label)
            idx += 1
            plt.imshow(image_numpy)
        plt.savefig("created/epoch"+str(epoch)+"img"+str(i)+".png")
        plt.close()


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_current_losses(epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)

