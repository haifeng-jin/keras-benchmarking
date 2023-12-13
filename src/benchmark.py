import sys

NUM_STEPS = 10
BERT_BATCH_SIZE = 8


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
