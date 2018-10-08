import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


array = np.load("generated_data.npy")
array = array.astype("float32")
data = []
for out_id in range(16):
    for image_id in range(len(array[out_id])):
        data.append((array[out_id, image_id], out_id))

np.random.seed(71)
np.random.shuffle(data)

test_num = int(len(data)*0.2)

test, train = data[:test_num], data[test_num:]

# Set up a neural network to train
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
n_unit = 1000
n_out = 16
model = L.Classifier(MLP(n_unit, n_out))

# Setup an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

batchsize = 100
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)

# Set up a trainer
epoch = 20
updater = training.updaters.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (epoch, 'epoch'), out="result")

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot for each specified epoch
frequency = 1
trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())


# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())


# Run the training
trainer.run()
