from config import load_args
from util.dataloader import multiscale_dataloader
from util.image_format import *
import torch
from torch.optim import lr_scheduler
from models import create_model
import matplotlib.pyplot as plt
from util import util


def train_layer(opt, dataloader, dataset):
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    model = create_model(opt, dataset.get_current_scale())
    total_iters = 0
    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            if epoch == opt.start_epoch and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt, None)
                model.parallelize()
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.visuals_snapshots == 0:
            # if total_iters % 99 == 0:
                model.compute_visuals()
                save_snapshot_visual(model.get_current_visuals(), epoch, i,
                                     opt.snapshot_dir, dataset.get_current_scale())

        model.update_learning_rate()  
        if epoch % opt.print_freq == 0:
            losses = model.get_current_losses()
            print_current_losses(epoch, 0, losses, 1, 1)
    model.save_networks(dataset.get_current_scale())
    if dataset.get_current_scale() != opt.num_scales:
        generate_next_scale(opt, model, dataset)
        dataset.increment_scale()


def generate_next_scale(opt, model, dataset):
    dataset.opt.serial_batches = True
    dataset.opt.no_flip = True
    preprocessing = dataset.opt.preprocess
    dataset.opt.preprocess = ""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_threads),
        drop_last=False,
    )

    scale = dataset.get_current_scale()
    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.eval()
        model.test()
        fake_B = model.get_current_visuals()["fake_B"]
        orig_A = model.get_current_visuals()["original_A"]
        save_scaled(opt, fake_B, orig_A, data["A_paths"], scale)

    dataset.opt.serial_batches = False
    dataset.opt.no_flip = False
    dataset.opt.preprocess = preprocessing


if __name__ == '__main__':
    # Train option parse
    arg_parser = load_args()
    opt = arg_parser.parse_args()
    util.set_gpu(opt)


    # create dataset
    dataset = multiscale_dataloader(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        drop_last=True if not opt.isTrain else False,
    )


    # add for loop for scales
    for scale in range(0, opt.num_scales):
        train_layer(opt, dataloader, dataset)
