#!/b

venvs=(
    "torch"
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)

for venv_name in "${venvs[@]}"; do
    if [ -d "$venv_name" ]; then
        if command -v deactivate &> /dev/null; then
            deactivate
        fi

        rm -rf "$venv_name"
        echo "Removed virtual environment: $venv_name"
    else
        echo "Virtual environment not found: $venv_name"
    fi
done
