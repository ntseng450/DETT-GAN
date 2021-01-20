from config import load_args
from util.dataloader import domain_dataloader


def train_layer(opt):


if __name__ == '__main__':
    arg_parser = load_args()
    opt = arg_parser.parse_args()
    dataset = multiscale_dataloader(opt)
    self.dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=opt.serial_batches,
        drop_last=True if opt.isTrain else False,
    )