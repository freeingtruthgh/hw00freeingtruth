# freeingtruth
freeingtruth is a Python package that currently provides:
- A discrete derivate for timeseries data
- A two-layer binary classification traning function.

# differential subpackage
This `differential` subpackage contains the function `diff`, which computes the discrete derivative of a time series.
## Despription
- Inputs: singal data and corresponding time data
- Output: discrete derivative of the timeseries.

## Installation
1. install `uv`
2.  install the package:
    ```bash
    pip install freeingtruth
    ```
    (dependencies should automatically install from inline metadata)

## run test code
A UV project named `difftest` exists is in `hw00freeingtruth` repository.
1. Clone the repository
2. Activate the virtual environment:
    ```bash
    cd difftest 
    source .venv/bin/activate
    ```
3. Run the Jupyter notebook:
    ```bash
    uv run difftest.ipynb
    ```
    A Jupyter kernel named `difftest` is also available for running the notebook.
4. Get Results:
    Inside `difftest` project directory there are results named Singal_vs_Time.pdf and Derivative_vs_Time.pdf. 

# deep1 subpackage
The `deep1` subpackage contains the function `binary_classification`, which trains a two-layer neural network for binary classification.

## Description
The function takes:
- `d`: number of features
- `n`: number of data samples
- `epochs`: number of training epochs (default: 10,000)
- `lr`: learning rate (default: 0.001)
It returns:
- Final trained weights
- Loss history at each epoch

## Installation (local build required)
The `deep1` subpackage is not included in the published wheel and must be built locally.
1. Clone the `hw00freeingtruth` repository
2. Navigate to the project directory:
    ```bash
    cd freeingtruth
    source .venv/bin/activate
    ```
3. Sync dependencies:
    ```bash
    uv sync
    ```

## Run test code
1. Navigate to the root directory of uv project and activate the virtual environment:
    ```bash
    cd freeingtruth
    source .venv/bin/activate
    ```

2. Run the test script:
    ```bash
    uv run src/scripts/binaryclassification_impl.py
    ```
3. Get Results:
    A plot should popup showing the loss vs epochs,, also inside `freeingtruth` project there will be a saved pdf with the name (YYYYMMDDhhmmss.pdf)

# Notes
- This package is under active development
- No updated `.whl` file is available for the `deep1` subpackage yet and requires a local build


