
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

    # encoding of string labels: "dog" -> 0, "cat" -> 1
    y = Category[filename.split('.')[0]].value

    # RGB image is flattened into a one long column vector of floats,
    # that denote color intensity
    x = np.array(resized, dtype=np.float64) \
          .reshape(width * height * 3, 1) / 256.

    return x, y, path


# load and preprocess images in parallel
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

    # calculate the values of partial derivatives
    dz = y_h - y
    dw = np.dot(x, dz.T) / m
    db = np.sum(dz) / m

    # update the weights for the next iteration
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


def logistic_regression(train_set, learning_rate=0.001, iterations=100, batch_size=64, callback=None):
    # stack training examples as columns into a (n, m) matrix
    x = np.column_stack(x[0] for x in train_set)
    y = np.array([x[1] for x in train_set], dtype=np.float64).reshape(1, len(train_set))

    # split the whole training set into batches of equal size
    n, m = x.shape
    num_batches = m // batch_size + (1 if m % batch_size > 0 else 0)
    x_batches = np.array_split(x, num_batches, axis=1)
    y_batches = np.array_split(y, num_batches, axis=1)

    # run the gradient descent to learn w and b
    w, b = initialize_weights(n)
    for iteration in range(iterations):
        j = 0

        for x_batch, y_batch in zip(x_batches, y_batches):
            y_hat = hypothesis(w, b, x_batch)
            w, b = update_weights(w, b, x_batch, y_hat, y_batch, learning_rate)

            j += cost(w, b, x_batch, y_batch) / num_batches

        if callback is not None:
            callback(iteration=iteration, w=w, b=b, cost=j)
    save(w,b)
    return w, b


def predict(w, b, x, threshold=0.5):
    y_h = hypothesis(w, b, x)
    return y_h >= threshold


def accuracy(w, b, data):
    # stack examples as columns into a (n, m) matrix
    x = np.column_stack(x[0] for x in data)
    y = np.array([x[1] for x in data]).reshape(1, x.shape[1])

    # calculate the accuracy value as a percentage of correct predictions
    correct_predictions = np.count_nonzero((y == predict(w, b, x)))
    total_predictions = x.shape[1]

    return correct_predictions / total_predictions

def save(w,b):
    np.save('weights.npy',w)
    np.save('biases.npy',b)
def main(path, iterations=500, max_examples=500, train_ratio=0.9):
    # load examples and make sure they are uniformly distributed
    examples = load_examples(path)
    print('number of ex',len(examples))
    random.shuffle(examples)

    # split all the examples into train and test sets
    m_train = int(max_examples * train_ratio)
    train_set = examples[:m_train]
    test_set = examples[m_train:]

    # monitor the progress of training
    def progress(iteration, cost, w, b):
        print('Iteration %d' % iteration)
        print('\tCost: %f' % cost)
        if iteration % 10 == 0:
            print('\tTrain set accuracy: %s' % accuracy(w, b, train_set))
            print('\tTest set accuracy: %s' % accuracy(w, b, test_set))

    # run the training process to learn model parameters
    w, b = logistic_regression(train_set, iterations=iterations, callback=progress)
    print('\tFinal train set accuracy: %s' % accuracy(w, b, train_set))
    print('\tFinal test set accuracy: %s' % accuracy(w, b, test_set))


main("C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\")
# main("test1")
