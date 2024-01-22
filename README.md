# DrumKitExtractor
Extract drum samples from a given audio file

# Work In Progress

## Python Version Requirement

This project requires Python 3.11 or higher. Ensure you have the correct Python version installed before running the code.

## CUDA Requirement

The current implementation of this project requires CUDA support. CUDA is a parallel computing platform and application programming interface (API) developed by NVIDIA for GPU acceleration. To utilize this project, you will need a compatible NVIDIA GPU and the necessary CUDA libraries and drivers installed on your system.

Please ensure that you have a CUDA-capable GPU and the corresponding CUDA toolkit and drivers installed to use this implementation effectively.

## Installation

In a python 3.11 environment (ex conda, venv)
```
pip install -r requirements.txt
```

## Command-Line Usage

To extract drum samples from an audio file, you can use the following command format:

```bash
python drum_extractor.py "PATH_OF_AUDIO_FILE"