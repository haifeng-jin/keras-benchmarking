import time

import keras

NUM_STEPS = 10

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
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput


def fit(model, dataset):
    callback = BenchmarkMetricsCallback(stop_batch=NUM_STEPS)
    model.fit(dataset, epochs=1, callbacks=[callback])
    print(f"fit: {1000.0 / callback.state['throughput']:.0f} ms/step")


def predict(model, dataset):
    callback = BenchmarkMetricsCallback(stop_batch=NUM_STEPS)
    model.predict(dataset, callbacks=[callback])
    print(f"predict: {1000.0 / callback.state['throughput']:.0f} ms/step")
