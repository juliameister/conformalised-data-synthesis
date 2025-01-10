import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
import seaborn as sns
sns.set_style("whitegrid")
import tensorflow as tf
from tensorflow.keras import (models, layers)


class FFNN():
    def __init__(self, input_shape, n_class, optimizer='adam', a1='relu', a_final='softmax', epochs=10, batch_size=64, validation_split=0.3, verbose=0, is_graph=False):
        self.input_shape, self.n_class, self.optimizer = input_shape, n_class, optimizer
        self.epochs, self.batch_size, self.validation_split, self.verbose, self.history = [None]*5
        self.a1, self.a_final = a1, a_final
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.is_graph = is_graph

        self.from_logits = not(self.a_final=='softmax' or self.a_final=='sigmoid')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits)

        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=self.input_shape))
        self.model.add(layers.Dense(32, activation=self.a1))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(16, activation=self.a1))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(8, activation=self.a1))
        self.model.add(layers.Dense(self.n_class, activation=self.a_final))
        self.model.compile(optimizer=self.optimizer,
                  loss = self.loss,
                  metrics=['accuracy'])

    def save(self, directory):
        self.model.save(directory)

    def load_model(self, directory):
        self.model = tf.keras.models.load_model(directory)

    def fit(self, X_train, y_train):
        tf.keras.backend.clear_session()
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_split=self.validation_split, verbose=self.verbose)
        if self.is_graph:
            self.eval_graph(figsize=(5, 2))
        return self

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

    def eval_graph(self, X_test=None, y_test=None, figsize=(10, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
        axs[0].plot(self.history.history['accuracy'], label="train")
        axs[0].plot(self.history.history['val_accuracy'], label="validation")
        axs[0].set(title='accuracy', ylabel='accuracy', xlabel='epoch')
        axs[0].legend(loc='lower right')

        axs[1].plot(self.history.history['loss'], label="train")
        axs[1].plot(self.history.history['val_loss'], label="validation")
        axs[1].set(title='loss', ylabel='loss', xlabel='epoch')
        axs[1].legend(loc='upper right')

        if not X_test is None:
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=self.verbose)
            axs[0].hlines(test_acc, 0, self.epochs-1, 'r', label="test")
            axs[0].legend(loc='lower right')
            axs[1].hlines(test_loss, 0, self.epochs-1, 'r', label="test")
            axs[1].legend(loc='upper right')
            return test_loss, test_acc
