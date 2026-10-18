# Magnetron Simulation

## Requirements

- [Miniconda](https://docs.anaconda.com/miniconda/)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jakubiee/Magnetron.git
cd Magnetron/simulation
```

### 2. Create the environment

Create the Conda environment and install all required dependencies:

```bash
conda env create -f environment.yml
conda activate magnetron
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open a notebook from the `notebooks/` directory and select the `magnetron` kernel if necessary.

## Usage

Simulation notebooks are located in `notebooks/`. Simulation outputs, including mesh files, field data and plots, are saved in `results/`.
