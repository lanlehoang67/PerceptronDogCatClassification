
import concurrent.futures
import enum
import multiprocessing
import os
import random
import sys
import numpy as np
from PIL import Image


class Category(enum.Enum):
    dog = 0
    cat = 1


def image_to_example(path, width=64, height=64):
    filename = os.path.basename(path)
    with Image.open(path) as img:
        resized = img.resize((width, height))
    y = Category[filename.split('.')[0]].value
    x = np.array(resized, dtype=np.float64) \
          .reshape(width * height * 3, 1) / 256.
    return x, y, path

def load_examples(path, width=64, height=64):
    concurrency = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as executor:
        images_futures = [
            executor.submit(
                image_to_example,
                os.path.join(path, name), width, height
            )
            for name in os.listdir(path)
        ]

        return [
            i.result()
            for i in concurrent.futures.as_completed(images_futures)
        ]

v1 = np.random.rand(10000, 1)
v2 = np.random.rand(10000, 1)


def initialize_weights(n):
    w = np.zeros((n, 1))
    b = 0.0
    return w, b


def hypothesis(w, b, x):
    z = np.dot(w.T, x) + b
    return 1. / (1. + np.exp(-z))


def cost(w, b, x, y):
    m = x.shape[1]
    y_h = hypothesis(w, b, x)

    return - np.sum(y * np.log(y_h) + (1 - y) * np.log(1 - y_h)) / m


def update_weights(w, b, x, y_h, y, learning_rate):
    m = x.shape[1]
    dz = y_h - y
    dw = np.dot(x, dz.T) / m
    db = np.sum(dz) / m
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


def logistic_regression(train_set, learning_rate=0.001, iterations=100, batch_size=64, callback=None):
    x = np.column_stack(x[0] for x in train_set)
    y = np.array([x[1] for x in train_set], dtype=np.float64).reshape(1, len(train_set))
    n, m = x.shape
    num_batches = m // batch_size + (1 if m % batch_size > 0 else 0)
    print('num',num_batches)
    print(m)
    x_batches = np.array_split(x, num_batches, axis=1)
    y_batches = np.array_split(y, num_batches, axis=1)
    w, b = initialize_weights(n)
    for iteration in range(iterations):
        j = 0

        for x_batch, y_batch in zip(x_batches, y_batches):
            y_hat = hypothesis(w, b, x_batch)
            w, b = update_weights(w, b, x_batch, y_hat, y_batch, learning_rate)

            j += cost(w, b, x_batch, y_batch) / num_batches

        if callback is not None:
            callback(iteration=iteration, w=w, b=b, cost=j)
    return w, b


def predict(w, b, x, threshold=0.5):
    y_h = hypothesis(w, b, x)
    return y_h >= threshold


def accuracy(w, b, data):
    x = np.column_stack(x[0] for x in data)
    y = np.array([x[1] for x in data]).reshape(1, x.shape[1])
    correct_predictions = np.count_nonzero((y == predict(w, b, x)))
    total_predictions = x.shape[1]

    return correct_predictions / total_predictions

def save(w,b):
    np.save('weights.npy',w)
    np.save('biases.npy',b)
def load():
    w = np.load('weights.npy')
    b = np.load('biases.npy')
    return w,b
def main(path, iterations=800, max_examples=25000, train_ratio=0.9):
    examples = load_examples(path)
    print('number of ex',len(examples))
    random.shuffle(examples)
    m_train = int(max_examples * train_ratio)
    train_set = examples[:m_train]
    test_set = examples[m_train:]
    def progress(iteration, cost, w, b):
        print('Iteration %d' % iteration)
        print('\tCost: %f' % cost)
        if iteration % 100 == 0:
            print('\tTrain set accuracy: %s' % accuracy(w, b, train_set))
            print('\tTest set accuracy: %s' % accuracy(w, b, test_set))
    w, b = logistic_regression(train_set, iterations=iterations, callback=progress)
    print('\tFinal train set accuracy: %s' % accuracy(w, b, train_set))
    print('\tFinal test set accuracy: %s' % accuracy(w, b, test_set))


# main("E:\\dog-or-cat\\dataset\\train\\training")
with Image.open('E:\\dog-or-cat\\dataset\\test\\testing\\92.jpg') as img:
    resized = img.resize((64, 64))
x = np.array(resized, dtype=np.float64) \
        .reshape(64 * 64 * 3, 1) / 256.
w,b = load()
print(predict(w,b,x))
#8,68,31,86,92