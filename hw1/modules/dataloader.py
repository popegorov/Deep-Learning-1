import math
import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.type = True
        if len(y.shape) == 1:
            self.type = False
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return math.ceil(self.X.shape[0]/self.batch_size)

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            indexes = np.arange(self.num_samples())
            np.random.shuffle(indexes)
            self.X = self.X[indexes]
            self.y = self.y[indexes]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if len(self) > self.batch_id + 1:
            x_batch = self.X[(self.batch_id * self.batch_size):
                             ((self.batch_id + 1) * self.batch_size)]
            y_batch = self.y[(self.batch_id * self.batch_size):
                             ((self.batch_id + 1) * self.batch_size)]
        elif len(self) == self.batch_id + 1:
            x_batch = self.X[(self.batch_id * self.batch_size):]
            y_batch = self.y[(self.batch_id * self.batch_size):]
        else:
            raise StopIteration

        self.batch_id += 1
        return x_batch, y_batch

