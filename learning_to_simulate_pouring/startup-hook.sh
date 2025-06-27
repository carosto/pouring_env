# Check if conda is available
if command -v conda &> /dev/null; then
    # Switch to the desired virtual environment
    eval "$(conda shell.bash hook)"
    conda activate myenv

    # Do that for every new interactive terminal session
    echo 'eval "$(conda shell.bash hook)" && conda activate myenv' >> ~/.bashrc
else
    echo "Conda is not installed in this environment."
fi