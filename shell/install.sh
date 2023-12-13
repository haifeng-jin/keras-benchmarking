#!/bin/bash

venvs=(
    "torch"
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)
requirements_files=(
    "requirements-torch.txt"
    "requirements-tensorflow.txt"
    "requirements-keras-tensorflow.txt"
    "requirements-keras-jax.txt"
    "requirements-keras-torch.txt"
)

for ((i=0; i<${#venvs[@]}; i++)); do
    venv_name=${venvs[$i]}
    req_file=${requirements_files[$i]}

    python -m venv ~/.venv/$venv_name
    source ~/.venv/$venv_name/bin/activate
    pip install -r requirements/$req_file
    deactivate

    echo "Installed libraries from $req_file in $venv_name"
done
