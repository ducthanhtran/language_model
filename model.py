import argparse
import os
import typing
import mxnet as mx

def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple LSTM language model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-layers', type=int, default=1, help='number of stacked layers')
    parser.add_argument('--num-hidden', type=int, default=128, help='hidden layer size of LSTM units')
    parser.add_argument('--num-embed', type=int, default=200, help='embedding layer size')
    return parser


class LanguageModelLSTM:
    """A simple (stacked) LSTM language model"""

    def __init__(self, num_layers: int, num_hidden_lstm: int) -> None:
        self.num_layers = num_layers
        self.num_hidden_lstm = num_hidden_lstm

        # Stacked RNN with LSTM cells
        rnn = mx.rnn.SequentialRNNCell()
        for i in range(self.num_layers):
            rnn.add(mx.rnn.LSTMCell(num_hidden=self.num_hidden_lstm, prefix='lstm_layer_%d'.format(i)))
        self.rnn = rnn

    def sym_gen(self, seq_length: int) -> :
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label_softmax')





if __name__ == '__main__':
    args = create_argparser().parse_args()
