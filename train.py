import argparse
from prepare_data import *
from model import *


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple LSTM language model on word level.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-layers', type=int, default=1, help='number of stacked layers')
    parser.add_argument('--num-hidden', type=int, default=128, help='hidden layer size of LSTM units')
    parser.add_argument('--num-embed', type=int, default=200, help='embedding layer size')
    return parser


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # TODO: VOCAB will be infered through data reading tool (to be developed)
    # lm = LanguageModelLSTM(args.num_layers, args.num_hidden_lstm, args.num_embed, VOCAB)

    # TODO: perform fitting/training by reading in training data and corresponding labels; context
    #       should be inferred from parameters and retrieved like in sockeye. Moreover,
    #       default_bucket_key has to be specified using training data iterator
    # mod = mx.module.BucketingModule(lm.sym_gen(), default_bucket_key = data_train.default_bucket_key)
    #
    # NOTE: default optimizer with SGD and learning rate of 0.01
    # mod.fit(train_data=,
    #         eval_data=,
    #         eval_metric=mx.metric.Perplexity(invalid_label), # default is 0
    #         ....
    #        )
