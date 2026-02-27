# PCCF: A Predictive Coding Contextual Framework for Adaptive AI

This repository contains the source code and experimental scripts for the paper **"PCCF: A Predictive Coding Contextual Framework for Adaptive Artificial Intelligence"**.

## Overview

The Predictive Coding Contextual Framework (PCCF) uses precision-weighted prediction error to modulate the update gain (or learning rate) of an online model, enabling rapid adaptation to concept drift while maintaining stability in stationary regimes.

This repository includes two main experiments:
1.  **Scalar Mean-Shift Task**: A controlled 1D drift detection and adaptation task.
2.  **Toy Language Model Task**: A next-token prediction task with an abrupt rule shift.

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### 1. Scalar Mean-Shift Experiment
Run the multi-seed evaluation (N=30 seeds):

```bash
python verify_pccf.py --mode multiseed --n-seeds 30 --out-prefix pccf_scalar_majorrev
```

This will generate:
- Summary statistics (`pccf_scalar_majorrev_summary.csv`)
- Visualization plots (`pccf_scalar_majorrev_*.png`)

### 2. Toy Language Model Rule-Shift Experiment
Run the multi-seed evaluation (N=20 seeds):

```bash
python verify_pccf_attention.py --task grammar --n-seeds 20 --out-prefix pccf_toylm_majorrev
```

This will generate:
- Summary statistics (`pccf_toylm_majorrev_summary.csv`)
- NLL and Accuracy plots (`pccf_toylm_majorrev_*.png`)

## Citation

If you use this code in your research, please cite our paper:

> Botao Wang, Xinnian Wang. "Simulating 'Vipassana': Regulating Precision Weights to Mitigate Rigidity and Hallucination in Large Language Models". *Cognitive Systems Research* (Draft).
