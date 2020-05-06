import tensorflow as tf
from tensorflow import keras

import numpy as np

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(
    128, activation='sigmoid'), keras.layers.Dense(10)])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

testLoss, testAcc = model.evaluate(test_images, test_labels, verbose=2)
print('\nAccuracy: ', testAcc)