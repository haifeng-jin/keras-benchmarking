import time
import benchmark

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments


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


dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=2,
)

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1.0,
    max_steps=benchmark.NUM_STEPS + 1,
)

timing_callback = TimingCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    callbacks=[timing_callback],
)

trainer.train()

# Calculate overall training time
overall_training_time = timing_callback.end_time - timing_callback.start_time
print("Overall Training Time:", overall_training_time, "seconds")


start_time = time.time()
trainer.predict(tokenized_datasets["test"].select(list(range((benchmark.NUM_STEPS + 1) * 8))))
end_time = time.time()
total_time = end_time - start_time

start_time = time.time()
trainer.predict(tokenized_datasets["test"].select(list(range(8))))
end_time = time.time()
total_time -= end_time - start_time

print("Overall Inferencing Time:", total_time, "seconds")
