import errno
import os
import typing

# Constants are from sockeye.constants.py
BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]


def prepare_data(training_file: str) -> :
    if not os.path.isfile(training_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), training_file)

    lines = open(training+data_file).readlines()
    mx.rnn.encode
