import os
from config import load_args
from util.dataloader import multiscale_dataloader
from util.image_format import *
from models import create_model
from util import util
from pathlib import Path


def generate_layer(opt, dataloader, dataset):
    model = create_model(opt, dataset.get_current_scale())

    for i, data in enumerate(dataloader):
        if i==0:
            model.data_dependent_initialize(data)
            model.setup(opt, dataset.get_current_scale())
            model.parallelize()
        model.set_input(data)
        model.eval()
        model.test()
        fake_B = model.get_current_visuals()["fake_B"]
        if dataset.get_current_scale() != opt.num_scales:
            save_scaled(opt, fake_B, model.get_current_visuals()["original_A"], data["A_paths"], dataset.get_current_scale())
        else:
            save_final(opt, fake_B, data["A_paths"])
    if dataset.get_current_scale() != opt.num_scales:
        dataset.increment_scale()


def save_final(opt, fake_B, A_path):
    output_dir = Path(opt.generate_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_B = Image.fromarray(tensor2im(fake_B))
    filename = os.path.basename(A_path[0])
    image_B.save(output_dir / Path(filename))


"""
Multi-Scale Unaligned Dataset
1. Initialize datasets and models
2. Run through the first scale and save generated 64x64 images
3. Run through the second scale and save generated 128x128 images
4. Run through the third scale and save generated 256x256 images
"""
if __name__ == '__main__':
    arg_parser = load_args()
    opt = arg_parser.parse_args()
    util.set_gpu(opt)
    opt.batch_size = 1
    opt.no_flip = True
    opt.isTrain = False
    opt.preprocess = ""
    dataset = multiscale_dataloader(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_threads),
        drop_last=True if not opt.isTrain else False,
    )

    for i in range(0, opt.num_scales):
        generate_layer(opt, dataloader, dataset)

