import argparse

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Dataloader params
    arg_parser.add_argument('--input_dir', default='datasets/road2icy')
    arg_parser.add_argument('--num_channels', default=3, type=int)
    arg_parser.add_argument('--num_scales', default=3, type=int)
    arg_parser.add_argument('--scale_factor', default=2, type=int)
    arg_parser.add_argument('--base_resolution', default=64, type=int)

    # Training params
    arg_parser.add_argument('--preprocess', type=str, default='none')
    arg_parser.add_argument('--noise_mult', default=0.15, type=float)
    arg_parser.add_argument('--no_flip', action='store_true')
    arg_parser.add_argument('--start_epoch', default=1, type=int)
    arg_parser.add_argument('--num_epochs', default=2, type=int)
    arg_parser.add_argument('--batch_size', default=1, type=int)
    arg_parser.add_argument('--serial_batches', action='store_true')

    # Testing Params
    arg_parser.add_argument('is_test', action='store_true')

    return arg_parser

