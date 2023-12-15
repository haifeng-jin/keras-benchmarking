#!/bin/bash


venv_path=~/.venv
venvs=(
    "torch"
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)

rm results.txt
for venv_name in "${venvs[@]}"; do
    echo "Running benchmarks for $venv_name." | tee -a results.txt
    source $venv_path/torch/bin/activate
    if [[ $venv_name == torch ]]; then
        file_name=torch
    else
        file_name=keras
    fi
    python src/bert_$file_name.py results.txt
    deactivate
done
