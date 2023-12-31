#!/bin/bash


venv_path=~/.venv
venvs=(
    "torch"
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)
output_file=results.txt

if [ -e "$output_file" ]; then
    rm -f "$output_file"
fi

export LD_LIBRARY_PATH=

for venv_name in "${venvs[@]}"; do
    echo "Benchmarking $venv_name" | tee -a $output_file
    source $venv_path/$venv_name/bin/activate
    if [[ $venv_name == torch ]]; then
        file_name=torch
    else
        file_name=keras
    fi

    if [[ $venv_name == tensorflow ]]; then
        export KERAS_HOME=configs/tensorflow
    fi

    if [[ $venv_name == keras* ]]; then
        export KERAS_HOME=configs/${venv_name#keras-}
    fi

    echo "Bert:" | tee -a $output_file
    python src/bert_$file_name.py $output_file

    echo "SegmentAnything:" | tee -a $output_file
    python src/sam_$file_name.py $output_file

    deactivate
done
