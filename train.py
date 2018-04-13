import argparse
from prepare_data import *
from model import *


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple LSTM language model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-layers', type=int, default=1, help='number of stacked layers')
    parser.add_argument('--num-hidden', type=int, default=128, help='hidden layer size of LSTM units')
    parser.add_argument('--num-embed', type=int, default=200, help='embedding layer size')
    return parser


if __name__ == '__main__':
    args = create_argparser().parse_args()
