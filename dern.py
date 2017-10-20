# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import progressbar


#torch.manual_seed(42)


def make_var(np_array, requires_grad=False):
    tensor = torch.from_numpy(np_array.astype(np.float32))
    return autograd.Variable(tensor, requires_grad=requires_grad)

class DERN(nn.Module):

    def __init__(self, feature_vec_dim):
        """
        """
        super(DERN, self).__init__()
        self.feature_vec_dim = feature_vec_dim
        self.linear = nn.Linear(feature_vec_dim, 1)
        self.loss_function = nn.BCELoss()

    def forward(self, X):
        """
        """
        return F.sigmoid(self.linear(X))


    def target(self, y_predicted, X_batch):
        """
        """
        y_predicted_np_array = y_predicted.data.numpy()
        target = np.random.randn(y_predicted_np_array.shape[0], 1).astype(np.float32)
        target[target>=0.5] = 1
        target[target<0.5] = 0
        """
            Тут надо посчитать значение target переменной (просто с error вычислительный граф становиться несвязным и autograd уже невозможно использовать)
        """
        return make_var(target)


    def train(self, batch_iterator, epochs=100, learning_rate=1e-3):
        """
        """
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        total_batches_cnt = batch_iterator.batches_cnt
        pbar = progressbar.ProgressBar(maxval=total_batches_cnt, widgets=[
                progressbar.DynamicMessage('Epoch'), # Static text
                ', ',
                progressbar.DynamicMessage('progress'),
                '%, ',
                progressbar.AdaptiveETA(),
                ', ',
                progressbar.DynamicMessage('loss'),
                ', ',
            ])
        for epoch in range(epochs):
            pbar.start()
            training_stat = {'Epoch': epoch + 1, 'progress': 0}
            for processed, X_batch in batch_iterator:
                self.zero_grad()
                X_batch_var = make_var(X_batch)
                y_predicted = self(X_batch_var)
                y_true = self.target(y_predicted, X_batch)

                loss = self.loss_function(y_predicted, y_true)
                loss.backward()
                optimizer.step()

                progress = 100.0*processed/total_batches_cnt
                training_stat.update({'progress': progress, 'loss':loss.data[0]})
                pbar.update(processed, **training_stat)
            pbar.finish()

    def predict(self, X):
        X_var = make_var(X)
        return self.forward(X_var).data.numpy()


class batch_iterator():

    def __init__(self):
        self.batches_cnt = 100
        self.cur_batch = -1

    def next(self):
        if self.cur_batch >= self.batches_cnt:
            self.cur_batch = -1
            raise StopIteration()
        self.cur_batch += 1
        bsize = np.random.randint(16, 32)
        return self.cur_batch, np.random.randn(32,1000)

    def __iter__(self): return self




if __name__ == '__main__':
    nn = DERN(1000)
    nn.train(batch_iterator(), epochs=10)
    X_test = np.random.rand(3, 1000)
    print nn.predict(X_test)
