from tqdm import trange
from answer import MultiLayerPerceptron
import numpy as np
import json
import pickle


class Trainer:
    def __init__(self,
                 network,
                 x_train,
                 t_train,
                 x_test,
                 t_test,
                 epochs=20,
                 batch_size=100,
                 learning_rate=0.001):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_size = x_train.shape[0]

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 勾配
        grad = self.network.gradient(x_batch, t_batch)

        # 更新
        for (i, dW, db) in grad:
            self.network.layers[i].W -= self.learning_rate*dW
            self.network.layers[i].b -= self.learning_rate*db

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
    trainer = Trainer(network, x_train, t_train, x_test, t_test)
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
