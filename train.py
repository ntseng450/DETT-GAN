from config import load_args
from util.dataloader import multiscale_dataloader
from util.image_format import tensor2im, save_image
import torch
from torch.optim import lr_scheduler

def train_layer(opt, dataloader):
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        for i, data in enumerate(dataloader):
            imgA = tensor2im(data["A"])
            save_file = "testing{}.png".format(i)
            save_image(imgA, save_file)
            print(i)


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
    train_layer(opt, dataloader)
    dataset.increment_scale()
