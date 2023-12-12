import sys
import time

import keras

NUM_STEPS = 10
BERT_BATCH_SIZE = 8


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
    callback = BenchmarkMetricsCallback(stop_batch=NUM_STEPS)
    model.fit(dataset, epochs=1, callbacks=[callback])
    return 1000.0 * callback.time_per_step


def predict(model, dataset):
    callback = BenchmarkMetricsCallback(stop_batch=NUM_STEPS)
    model.predict(dataset, callbacks=[callback])
    return 1000.0 * callback.time_per_step


def append_to_file(file_path, content):
    try:
        with open(file_path, "a") as file:
            file.write(content + "\n")
        print(f"Content appended to {file_path}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


def benchmark(run):
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        training_per_step, inferencing_per_step = run()
        content = (
            f"Training: {training_per_step} ms/step\n"
            f"Inferencing: {inferencing_per_step} ms/step\n"
        )
        print(content)
        file_path = sys.argv[1]
        append_to_file(file_path, content)
