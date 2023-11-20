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
        batch_size=8,
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


dataset = get_dataset().take(101)
model = get_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adamw",
)

benchmark.fit(model, dataset)
benchmark.predict(model, dataset)
