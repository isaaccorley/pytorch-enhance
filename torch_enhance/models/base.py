import torch.nn as nn


class Base(object):
    """
    Base SR module containing common methods
    """
    def normalize01(self, x):
        """ Normalize from [0, 255] -> [0, 1] """
        return x / 255

    def denormalize01(self, x):
        """ Normalize from [0, 1] -> [0, 255] """
        return x * 255

    def normalize11(self, x):
        """ Normalize from [0, 255] -> [-1, 1] """
        return x / 127.5 - 1

    def denormalize11(self, x):
        """ Normalize from [-1, 1] -> [0, 255] """
        return (x + 1) * 127.5
