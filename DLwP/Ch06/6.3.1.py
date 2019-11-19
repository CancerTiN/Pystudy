import logging

import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fname = r'D:\Workspace\Study\DLwP\Ch06\jena_climate_2009_2016.csv'
with open(fname) as handle:
    header = handle.readline().split(',')
    lines = handle.readlines()
logger.info(header)
logger.info(len(lines))

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.waitforbuttonpress()
plt.close()
plt.plot(range(1440), temp[:1440])
plt.waitforbuttonpress()
plt.close()

mean = float_data[:20000].mean(axis=0)
float_data -= mean
std = float_data[:20000].std(axis=0)
float_data /= std


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback, delay, 0, 200000, True, batch_size)
val_gen = generator(float_data, lookback, delay, 200001, 300000, False, batch_size)
test_gen = generator(float_data, lookback, delay, 200001, None, False, batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size


def evaluate_naive_method():
    batch_maes = list()
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)


mae = evaluate_naive_method()
logger.info(mae)
celsius_mae = mae * std[1]
logger.info(celsius_mae)
