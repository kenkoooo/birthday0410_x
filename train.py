from tqdm import trange
from answer import MultiLayerPerceptron
import numpy as np
import json
import pickle
from typing import Dict


class Trainer:
    def __init__(self,
                 network,
                 x_train,
                 t_train,
                 x_test,
                 t_test,
                 optimizer,
                 epochs=20,
                 batch_size=100):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.train_size = x_train.shape[0]

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 勾配
        grad_list = self.network.gradient(x_batch, t_batch)

        params = {}
        grads = {}

        # update
        for (i, dW, db) in grad_list:
            params["{}_W".format(i)] = self.network.layers[i].W
            params["{}_b".format(i)] = self.network.layers[i].b
            grads["{}_W".format(i)] = dW
            grads["{}_b".format(i)] = db

        self.optimizer.update(params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

    def train(self):
        with trange(self.epochs) as epoch_range:
            for _ in epoch_range:
                epoch_size = self.train_size//self.batch_size
                with trange(epoch_size) as train_range:
                    sum_loss = 0.0
                    for _ in train_range:
                        self.train_step()

                        loss = self.train_loss_list[-1]
                        sum_loss += loss
                        train_range.set_postfix(loss=loss)

                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)

                epoch_range.set_postfix(
                    train_acc=train_acc,
                    test_acc=test_acc,
                    average_loss=sum_loss/epoch_size)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 **
                                 self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


def load_images(normalize=True, flatten=True, one_hot_label=False, test_ratio=0.8):
    np.random.seed(71)
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    if not normalize:
        images = images * 255
    if one_hot_label:
        n_labels = len(labels)
        one_hot_labels = np.zeros((n_labels, 16))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        labels = one_hot_labels

    count = len(images)
    data = []
    for i in range(count):
        data.append((images[i], labels[i]))

    np.random.shuffle(data)
    train_num = int(count * test_ratio)
    train, test = data[:train_num], data[train_num:]

    x_train = np.array([image for image, label in train])
    t_train = np.array([label for image, label in train])
    x_test = np.array([image for image, label in test])
    t_test = np.array([label for image, label in test])

    return (x_train, t_train), (x_test, t_test)


def main():
    (x_train, t_train), (x_test, t_test) = load_images(
        normalize=True, one_hot_label=True)
    print("x_train.shape", x_train.shape)
    print("t_train.shape", t_train.shape)
    print("x_test.shape", x_test.shape)
    print("t_test.shape", t_test.shape)

    n_in = x_train.shape[1]
    n_out = t_train.shape[1]
    network = MultiLayerPerceptron(
        n_in=n_in, n_units=1000, n_out=n_out)
    optimizer = Adam()
    trainer = Trainer(network, x_train, t_train, x_test, t_test, optimizer)
    trainer.train()

    with open("train_acc.json", "w") as f:
        json.dump(trainer.train_acc_list, f)
    with open("test_acc.json", "w") as f:
        json.dump(trainer.test_acc_list, f)
    with open("train_loss.json", "w") as f:
        json.dump(trainer.train_loss_list, f)

    with open("network.pickle", mode="wb") as f:
        pickle.dump(network, f)


if __name__ == "__main__":
    main()
