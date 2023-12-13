import time

import keras

import benchmark


class BenchmarkMetricsCallback(keras.callbacks.Callback):
    def __init__(self, start_batch=1, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        self.state = {}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            self.time_per_step = (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            ) / (self.stop_batch - self.start_batch + 1)

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            self.time_per_step = (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            ) / (self.stop_batch - self.start_batch + 1)


def fit(model, dataset):
    callback = BenchmarkMetricsCallback(stop_batch=benchmark.NUM_STEPS)
    model.fit(dataset, epochs=1, callbacks=[callback])
    return 1000.0 * callback.time_per_step


def predict(model, dataset):
    callback = BenchmarkMetricsCallback(stop_batch=benchmark.NUM_STEPS)
    model.predict(dataset, callbacks=[callback])
    return 1000.0 * callback.time_per_step
