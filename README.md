# FWEG

This contains the source code for the FWEG method. The `src/ipynb` folder contains most of the relevant code. The directory outline is showed below.

## Repository Overview

```
src/ipynb
│───adult
│───adult_bb
│───adult_periodic
│───cifar10
│───adience
│───adience_ablation
```

Each folder is a particular kind of experiment. **After setting up your machine and environment (see [Setup](##Setup)), you can quickly get started by navigating to the `adult` or `cifar10` folders and running the files in the following order:**

```
process.ipynb
train.ipynb
fweg.ipynb
# optional: run the baselines
run_baselines.ipynb
evaluate_baselines.ipynb
non_model_baselines.ipynb
```

In general, the order to run files can be found in `src/drive.py:EXPERIMENT_TO_FILE_ORDER`. Experiment configurations can be found in `src/config.json`. There you can fix the experiment random seed and some important experiment-specific settings. If you'd like to run everything as a Python script instead of a Jupyter notebook, navigate to the root of the repository and run `make py`. This uses `ipython nbconvert` to convert all the Jupyter notebooks into Python files. They are saved to `src/py` and mirror `src/ipynb`. Then, edit `src/drive.py` to configure your script run. `src/drive.py` determines which experiments and files are run and how they are configured using `config.json`. Please see it for more details on how to use it. Finally, simply run `python src/drive.py`.

## Setup

### Requirements

Install dependencies with `pip install -r requirements.txt`. You can optionally work in a Python virtual environment.

### Data

`adult` and `cifar10` are the easiest to run and require no setup. `adience` requires downloading data from https://talhassner.github.io/home/projects/Adience/Adience-data.html (specifically `faces.tar.gz`). It should be extracted and placed into `data/adience`. The folder should look as follows:

```
data/adience
│───faces
│───faces.tar.gz # the originally downloaded file; after extraction the other files listed here are made
│───fold_0_data.txt
│───fold_1_data.txt
│───fold_2_data.txt
│───fold_3_data.txt
│───fold_4_data.txt
```

Because the code is in Jupyter notebooks, it should be clear which step breaks and (hopefuly) why. Please open an issue for anything that breaks.

### Machine Requirements

We use UMAP to reduce dimensionality of high-dimensional data like images. This is used in CIFAR and adience. The memory requirement is upwards
of 16 GB. You can use swapfiles to get around memory constraints. This might be helpful: https://linuxize.com/post/create-a-linux-swap-file/.

Also, we highly recommend using a machine with a CUDA-enabled GPU. We used a single NVIDIA P100 GPU core for this entire project.

## Paper Configurations

This details specific configurations used in the paper. `src/drive.py` should already iterate over these configurations. `src/config.json` chooses a specific configuration and can be manually modified.

Random seeds:

- 15, 25, 35, 45, 55

Cifar:

- Asymmetric noise with 0.6 noise probability

Adience Ablation:

- Val split: [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
