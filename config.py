import argparse
import inspect

import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 15
    batch_size = 128
    learning_rate = 0.002
    l2_regularization = 1e-6
    learning_rate_decay = 0.99

    train_file = 'data/music/train.csv'
    valid_file = 'data/music/valid.csv'
    test_file = 'data/music/test.csv'
    model_file = 'model/best_model.pt'

    extension = False  # Use TransNet-Ext or not
    user_count = 5541  # Parameter of TransNet-EXT. Got from output of preprocess.py
    item_count = 3568

    review_count = 10  # max review count
    review_length = 80  # max review length
    lowest_review_count = 2  # reviews wrote by a user/item will be delete if its amount less than such value.
    PAD_WORD = '<UNK>'
    kernel_count = 100
    kernel_size = 3
    dropout_prob = 0.5
    cnn_out_dim = 50

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
