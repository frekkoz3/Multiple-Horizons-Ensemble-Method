# Multiple Horizons Ensemble Method (MHEMe)

This repository contains the material for the final project of the **Statistical Methods** course (2025/2026).

## Team

- Bortolussi Lorenzo  
- Bredariol Francesco  
- Savorgnan Enrico  

## Project Overview

The goal of this project is to develop **MHEMe (Multiple Horizons Ensemble Method)**, a novel ensemble approach for **time series forecasting** aimed at reducing prediction variance.

The method combines forecasts generated at multiple prediction horizons into a single ensemble prediction, leveraging horizon diversity to improve stability and robustness.

## Repository Structure

```bash
├─ data             # data used for evaluation of the method
├─ experiments      # folder containing different tests
├─ report           # some reports
├─ results          # folder containing results over benchmarks
├─ literature       # folder containing literature research about the project
├─ src              # method's Python implementation
└─ models           # folder containing models' saving files
```

## Results

The proposed method is evaluated on standard time series datasets and compared against baseline forecasting models in terms of variance and predictive accuracy.

## Quick setup

To run the code contained in this repository, please follow the instructions below.

```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .
    nbdime config-git --enable
    git lfs install
```

> **Note**
If you are having trouble with some files showing up as git lfs pointer, just copy and paste the following inastructions into the terminal:

```bash
    git lfs install
    git lfs pull
```

## License

MIT License
