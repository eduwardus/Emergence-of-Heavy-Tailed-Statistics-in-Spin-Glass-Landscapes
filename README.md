# Emergence-of-Heavy-Tailed-Statistics-in-Spin-Glass-Landscapes
# M19 - Spin Glass Landscape Analysis

This repository contains the full computational pipeline and results for the analysis of energy landscapes in spin glass systems under varying kurtosis (κ).

## Main result

Unlike Graph Laplacian Geometry (M18), spin glass systems do not exhibit a universal structural phase transition. Instead, kurtosis affects the statistical distribution of energy gaps, leading to heavy-tailed behavior.

## Pipeline

1. Generate spin glass instances
2. Compute energy landscapes
3. Attempt sigmoid collapse (negative result)
4. Structural analysis
5. Gap distribution analysis
6. Tail fitting (power law vs log-normal)

## Reproducibility

Run:

```bash
bash reproducibility/run_all.sh
