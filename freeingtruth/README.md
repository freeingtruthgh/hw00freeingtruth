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

# deepl subpackage
The `deepl` subpackage contains the function `binary_classification`, which trains a two-layer neural network for binary classification.

## Description
The function takes:
- `d`: number of features
- `n`: number of data samples
- `epochs`: number of training epochs (default: 10,000)
- `lr`: learning rate (default: 0.001)
It returns:
- trained weights at each epoch
- Loss history at each epoch

## Installation (local build required)
The `deepl` subpackage is not included in the published wheel and must be built locally.
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
    uv run scripts/binaryclassification_impl.py
    ```
3. Get Results:
    A plot should popup showing the loss vs epochs,, also inside `freeingtruth` project there will be a saved pdf with the name (YYYYMMDDhhmmss.pdf)

# HW02Q7
A new subpackage has been added called animation with functions to anime stacked matrices. A test script has been made in the `scripts` folder called `binaryclassification_animate_impl.py`
## Running the test script
1. Navigate to the project directory:
    ```bash
    cd freeingtruth
    source .venv/bin/activate
    ```
2. Sync dependencies and build:
    ```bash
    uv sync
    uv build
    ```
3. Run the test script:
    ```bash
    cd scripts
    uv run binaryclassification_animate_impl.py
    ```
    Alternatively, you can run in the background with:
    ```bash
    cd scripts
    nohup ./binary_animation.sh > training_log.out 2>&1 &
    ```
3. Get Results:
    Inside `scripts` project there will be a saved pdf with the name (YYYYMMDDhhmmss.pdf) this is the plot of loss vs epochs. Inside `media` under `videos` there are four .mp4 files, these are animations showing the evolution of the 4 weights over training epochs.

# HW02Q8
A new file has been added to deepl called `multiclass.py`  with classes to define a simple neural network and train. A test script has been made in the `scripts` folder called `multiclass_impl.py` which saves a csv file with final values for training and testing to evaluate the SimpleNN. As well `multiclass_eval.py` takes the csv file and creates a boxplot. `multiclass_impl.sh` runs 5 experiments with the same parameter values for `multiclass_impl.py` and then runs `multiclass_eval.py`
## Running the test script
1. Navigate to the project directory:
    ```bash
    cd freeingtruth
    source .venv/bin/activate
    ```
2. Sync dependencies and build (if needed):
    ```bash
    uv sync
    uv build
    ```
3. Run the test script:
    ```bash
    cd scripts
    ./multiclass_impl.sh
    ```
3. Get Results:
    Inside `scripts` there will be a saved pdf with the name ({tag}_YYYYMMDDhhmmss.pdf) this is the boxplot of final accuracy, F1 score, precision, and recall for training/test metrics as well there will be multiple CSV files with a similar naming convention.

# Notes
- This package is under active development
- No updated `.whl` file is available for the `deepl` subpackage yet and requires a local build


