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
        .repeat(benchmark.NUM_STEPS + 1)
        .batch(benchmark.SAM_BATCH_SIZE)
    )


def get_model():
    return keras_cv.models.SegmentAnythingModel.from_preset("sam_huge_sa1b")


def run():
    dataset = get_dataset()
    model = get_model()
    return None, keras_utils.predict(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
