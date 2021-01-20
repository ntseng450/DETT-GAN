import argparse

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Dataloader params
    arg_parser.add_argument('--input_dir', default='datasets/road2icy')
    arg_parser.add_argument('--num_scales', default=3, type=int)
    arg_parser.add_argument('--scale_factor', default=2, type=int)
    arg_parser.add_argument('--base_resolution', default=64, type=int)

    return arg_parser

