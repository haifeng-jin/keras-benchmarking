#!/b

venvs=(
    "torch"
    "tensorflow"
    "keras-tensorflow"
    "keras-jax"
    "keras-torch"
)

# Loop through each virtual environment
for venv_name in "${venvs[@]}"; do
# Check i
if [ -d "$venv_name" ]; then
# Deactivate the virtual environment if it's currently active
if command -v deactivate &> /dev/null; then
deactivate
fi
# Remove the virtual environment
rm -rf "$venv_name"
echo "Removed virtual environment: $venv_name"
else
echo "Virtual environment not found: $venv_name"
fi
done
