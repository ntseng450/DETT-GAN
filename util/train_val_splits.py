import argparse
import os
import numpy as np
from pathlib import Path
import shutil

def load_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--input_dir', default='datasets/road2icy/original')
    arg_parser.add_argument('--output_dir', default='test')
    arg_parser.add_argument('--size_val', type=int, default=230)

    return arg_parser


if __name__ == '__main__':
    arg_parser = load_args()
    opt = arg_parser.parse_args()

    # Get domain A
    A_path = opt.input_dir / Path("trainA")
    num_files = 0
    for root, _, fnames in sorted(os.walk(A_path, followlinks=True)):
        for fname in sorted(fnames):
            num_files += 1
    print(num_files)

    output_dir = opt.input_dir / Path(opt.output_dir+'A')
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    random_indices = np.random.choice(num_files, opt.size_val, replace=False) 
    for root, _, fnames in sorted(os.walk(A_path, followlinks=True)):
        for fname in sorted(fnames):
            if counter in random_indices:
                curr_path = os.path.join(root, fname)
                new_path = opt.input_dir / Path(opt.output_dir+'A') / fname
                shutil.move(curr_path, new_path)
            counter += 1

    # Get domain B
    B_path = opt.input_dir / Path("trainB")
    num_files = 0
    for root, _, fnames in sorted(os.walk(B_path, followlinks=True)):
        for fname in sorted(fnames):
            num_files += 1
    print(num_files)

    output_dir = opt.input_dir / Path(opt.output_dir+'B')
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    random_indices = np.random.choice(num_files, opt.size_val, replace=False) 
    for root, _, fnames in sorted(os.walk(B_path, followlinks=True)):
        for fname in sorted(fnames):
            if counter in random_indices:
                curr_path = os.path.join(root, fname)
                new_path = opt.input_dir / Path(opt.output_dir+'B') / fname
                shutil.move(curr_path, new_path)
            counter += 1
        
