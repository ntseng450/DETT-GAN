from config import load_args
from util.dataloader import domain_dataloader

arg_parser = load_args()
opt = arg_parser.parse_args()
domain_dataloader(opt)