import errno
import os
import typing
import mxnet as mx

class LanguageModelLSTM:
    def __init__(self, num_layers: int=1, num_hidden_lstm: int=128) -> None:
        self.num_layers = num_layers
        self.num_hidden_lstm = num_hidden_lstm

    def sym_gen(seq_length):
        mx.sym.
        # Stacked RNN with LSTM cells
        rnn = mx.rnn.SequentialRNNCell()
        for i in range(num_layers):
            rnn.add(mx.rnn.LSTMCell(num_hidden=num_hidden_lstm, prefix='lstm_layer%d'.format(i)))
        self.rnn = rnn

        self.hidden_weight = mx.symb.Variable('hidden_weight')
        self.hidden_bias = mx.symb.Variable('hidden_bias')
