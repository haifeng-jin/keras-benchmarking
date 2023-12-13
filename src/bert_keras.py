import keras
import keras_nlp
import tensorflow_datasets as tfds

import benchmark


# Make sure our optimization strategy is the same.
# In particular AdamW with exluded terms from the bias decay.
class BertOptimizer(keras.optimizers.AdamW):
    def __init__(
        self,
        learning_rate=5e-5,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        global_clipnorm=1.0,
        exclude_bias_from_decay=True,
        exclude_layernorm_from_decay=True,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            global_clipnorm=global_clipnorm,
        )
        if exclude_bias_from_decay:
            self.exclude_from_weight_decay(var_names=["bias"])
        if exclude_layernorm_from_decay:
            self.exclude_from_weight_decay(var_names=["gamma"])
            self.exclude_from_weight_decay(var_names=["beta"])


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
        optimizer=BertOptimizer(),
    )

    return benchmark.fit(model, dataset), benchmark.predict(model, dataset)


if __name__ == "__main__":
    benchmark.benchmark(run)
