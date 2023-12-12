import time

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

import benchmark


class TimingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        # Record start time only once at the beginning of the second step
        # Steps are [0, 100].
        if state.global_step == 1 and self.start_time is None:
            self.start_time = time.time()
        super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        # Record end time at the end of the last step
        # Steps are [0, 100].
        if state.global_step == benchmark.NUM_STEPS:
            self.end_time = time.time()


def get_dataset():
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True
        ),
        batched=True,
    )

    return tokenized_datasets


def run():
    dataset = get_dataset()
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        per_device_train_batch_size=benchmark.BERT_BATCH_SIZE,
        per_device_eval_batch_size=benchmark.BERT_BATCH_SIZE,
        num_train_epochs=1.0,
        max_steps=benchmark.NUM_STEPS + 1,
    )

    timing_callback = TimingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        callbacks=[timing_callback],
    )

    trainer.train()

    # Calculate overall training time
    overall_training_time = (
        timing_callback.end_time - timing_callback.start_time
    )
    training_per_step = overall_training_time / benchmark.NUM_STEPS * 1000

    start_time = time.time()
    trainer.predict(
        dataset["test"].select(list(range((benchmark.NUM_STEPS + 1) * 8)))
    )
    end_time = time.time()
    total_time = end_time - start_time

    start_time = time.time()
    trainer.predict(dataset["test"].select(list(range(8))))
    end_time = time.time()
    total_time -= end_time - start_time

    inferencing_per_step = total_time / benchmark.NUM_STEPS * 1000
    return training_per_step, inferencing_per_step


if __name__ == "__main__":
    benchmark.benchmark(run)
