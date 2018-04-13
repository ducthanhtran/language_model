import mxnet as mx

class LanguageModelLSTM:
    """A simple (stacked) LSTM language model"""

    def __init__(self, num_layers: int, num_hidden_lstm: int, vocab_size: int, embedding_dim: int) -> None:
        """
        Read in required parameters regarding LSTM network structure as well as vocabulary and embedding
        dimensionality.

        :param num_layers: number of stacked RNN layers (with LSTM cells)
        :param num_hidden_lstm: number of hidden units within each LSTM cell
        :param vocab_size: vocabulary size
        :param embedding_dim: vector dimensionality of word embeddings
        """
        self.num_layers = num_layers
        self.num_hidden_lstm = num_hidden_lstm
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Stacked RNN with LSTM cells
        rnn_stack = mx.rnn.SequentialRNNCell()
        for i in range(self.num_layers):
            rnn_stack.add(mx.rnn.LSTMCell(num_hidden=self.num_hidden_lstm, prefix='lstm_layer_%d'.format(i)))
        self.rnn_stack = rnn_stack

        # weight parameters for neural network
        self.weight_embed = mx.sym.Variable('weight_embed')

    def sym_gen(self, seq_length: int):
        """
        Creates a new computation graph by unrolling the stacked RNN with LSTM cells
        according to a parameter 'seq_length'. Therefore, different unrolling lengths can be acquired
        and training can be done more efficiently by using bucketing strategies.

        :param seq_length: length of sequence; states how many steps in time we unroll our stacked RNN for
        """
        # shape: (batch_size, 1)
        label = mx.sym.Variable('label')
        # shape: (batch_size, seq_length)
        data = mx.sym.Variable('data')
        # shape: (batch_size, seq_length, self.embedding_dim)
        embedding = mx.sym.Embedding(data=data, weight=self.weight_embed
                                     input_dim=self.vocab_size, output_dim=self.embedding_dim)

        # NOTE: mx.rnn.SequentialRNNCell.unroll() does a reset as well - our reset command might not be needed
        self.rnn_stack.reset()
        outputs = self.rnn_stack.unroll(length=seq_length, inputs=embedding, merge_outputs=True)[0]

        # softmax output layer
        pred = mx.sym.reshape(data=outputs, shape=(-1, self.num_hidden_lstm))

        pred = mx.sym.FullyConnected(data=pred, num_hidden=self.vocab_size, name='softmax_pred')
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return (sm, ['data'], ['label'])
