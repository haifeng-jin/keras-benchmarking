import keras
import keras_cv
import numpy as np
import tensorflow as tf

import benchmark
import keras_utils


def get_dataset():
    if keras.backend.image_data_format() == "channels_last":
        images = np.random.rand(1, 1024, 1024, 3)
    else:
        images = np.random.rand(1, 3, 1024, 1024)

    data = {
        "images": images,
        "points": np.array([[[500, 375], [250, 375]]]),
        "labels": np.array([[1, 2]]),
    }
    return (
        tf.data.Dataset.from_tensor_slices(data)
        .repeat((benchmark.NUM_STEPS + 1) * benchmark.SAM_BATCH_SIZE)
        .batch(benchmark.SAM_BATCH_SIZE)
    )


def get_train_dataset():
    if keras.backend.image_data_format() == "channels_last":
        images = np.random.rand(1, 1024, 1024, 3)
        features = np.random.rand(1, 64, 64, 256)
    else:
        images = np.random.rand(1, 3, 1024, 1024)
        features = np.random.rand(1, 256, 64, 64)

    return (
        tf.data.Dataset.from_tensor_slices((images, features))
        .repeat((benchmark.NUM_STEPS + 1) * benchmark.SAM_BATCH_SIZE)
        .batch(benchmark.SAM_BATCH_SIZE)
    )


def get_model():
    return keras_cv.models.SegmentAnythingModel.from_preset("sam_base_sa1b")


def run():
    train_dataset = get_train_dataset()
    dataset = get_dataset()
    model = get_model()
    backbone = model.backbone
    backbone.compile(loss="mse", optimizer="adam")
    return keras_utils.fit(backbone, train_dataset), keras_utils.predict(
        model, dataset
    )


if __name__ == "__main__":
    benchmark.benchmark(run)
