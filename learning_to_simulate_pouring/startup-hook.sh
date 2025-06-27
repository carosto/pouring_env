# Check if conda is available
if command -v conda &> /dev/null; then
    # Switch to the desired virtual environment
    eval "$(conda shell.bash hook)"
    conda activate myenv

    # Do that for every new interactive terminal session
    echo 'eval "$(conda shell.bash hook)" && conda activate myenv' >> ~/.bashrc

    # Install required Python packages
    pip gymnasium==1.1.1
    pip install /server/pouring_env/ 
else
    echo "Conda is not installed in this environment."
fi