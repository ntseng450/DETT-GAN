from config import load_args
from util.dataloader import multiscale_dataloader
from util.image_format import tensor2im, save_image
import torch
from torch.optim import lr_scheduler
from models import create_model
import matplotlib.pyplot as plt
from util import util


def train_layer(opt, dataloader):
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    model = create_model(opt)
    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        for i, data in enumerate(dataloader):
            if epoch == opt.start_epoch and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
            model.set_input(data)
            model.optimize_parameters()

        if epoch % opt.epoch_snapshots == 0:
            model.compute_visuals()
            display_current_results(model.get_current_visuals(), epoch, i, epoch)

        if epoch % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            # visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
            print_current_losses(epoch, 0, losses, 1, 1)


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

    print(message)  # print the message
    # with open(self.log_name, "a") as log_file:
    #     log_file.write('%s\n' % message)  # save the message


def display_current_results(visuals, epoch, i, save_result):
    fig = plt.figure()
    ncols = 4
    if ncols > 0:        # show all the images in one visdom panel
        ncols = min(ncols, len(visuals))
        idx = 0
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            subplot_fig = fig.add_subplot(1, ncols, idx+1)
            subplot_fig.set_title(label)
            idx += 1
            plt.imshow(image_numpy)
        plt.savefig("created/"+str(epoch)+"_epoch"+str(i)+".png")
        plt.close()


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


if __name__ == '__main__':
    arg_parser = load_args()
    opt = arg_parser.parse_args()
    dataset = multiscale_dataloader(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.serial_batches,
        drop_last=True if not opt.is_test else False,
    )
    print(len(dataset))

    # add for loop for scales
    dataset.increment_scale()
    dataset.increment_scale()
    train_layer(opt, dataloader)
