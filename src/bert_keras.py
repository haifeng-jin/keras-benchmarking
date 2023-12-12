import keras
import keras_nlp
import tensorflow_datasets as tfds

import benchmark


def get_dataset():
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_base_en",
    )

    dataset = tfds.load(
        "imdb_reviews",
        split="train",
        as_supervised=True,
        batch_size=benchmark.BERT_BATCH_SIZE,
    )
    # Force the dataset to cache into memory.
    dataset = dataset.map(preprocessor).cache()
    count = 0
    for batch in dataset:
        count += 1
    return dataset


def get_model():
    return keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en",
        num_classes=2,
        preprocessor=None,
    )


def run():
    dataset = get_dataset().take(benchmark.NUM_STEPS + 1)
    model = get_model()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adamw",
    )

    return benchmark.fit(model, dataset), benchmark.predict(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
