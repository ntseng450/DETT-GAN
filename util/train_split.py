import argparse
import os
import numpy as np
from pathlib import Path
import shutil

def load_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--input_dir', default='cmvid2icy/')
    arg_parser.add_argument('--create_original', action='store_true')
    arg_parser.add_argument('--size', type=int, default=150)

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

    if opt.create_original:
        output_dir = Path(opt.input_dir+ "_"+ str.opt.size) / Path("original")
    else:
        output_dir = Path(opt.input_dir + "_" + str(opt.size))
    output_dir.mkdir(parents=True, exist_ok=True)
    trainA_path = output_dir / Path("trainA")
    trainA_path.mkdir(parents=True, exist_ok=True)
    counter = 0
    random_indices = np.random.choice(num_files, opt.size, replace=False) 
    for root, _, fnames in sorted(os.walk(A_path, followlinks=True)):
        for fname in sorted(fnames):
            if counter in random_indices:
                curr_path = os.path.join(root, fname)
                new_path = output_dir / Path("trainA") / fname
                shutil.move(curr_path, new_path)
            counter += 1

    # Get domain B
    B_path = opt.input_dir / Path("trainB")
    num_files = 0
    for root, _, fnames in sorted(os.walk(B_path, followlinks=True)):
        for fname in sorted(fnames):
            num_files += 1
    print(num_files)

    if opt.create_original:
        output_dir = Path(opt.input_dir+ "_"+ str.opt.size) / Path("original")
    else:
        output_dir = Path(opt.input_dir + "_" + str(opt.size))
    output_dir.mkdir(parents=True, exist_ok=True)
    trainB_path = output_dir / Path("trainB")
    trainB_path.mkdir(parents=True, exist_ok=True)
    counter = 0
    random_indices = np.random.choice(num_files, opt.size, replace=False) 
    for root, _, fnames in sorted(os.walk(B_path, followlinks=True)):
        for fname in sorted(fnames):
            if counter in random_indices:
                curr_path = os.path.join(root, fname)
                new_path = output_dir / Path("trainB") / fname
                shutil.move(curr_path, new_path)
            counter += 1