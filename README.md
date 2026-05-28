# BML Traffic Model Simulation

Parallel simulation framework for studying the Biham–Middleton–Levine (BML) traffic model and emergent phase-transition behavior in lattice-based systems.

## Overview

The BML model is a cellular automaton used to study self-organization, congestion, and phase transitions in traffic-like systems.
This project explores how simple local update rules generate large-scale collective behavior.

## Features

* Parallelized simulation framework in C++
* Configurable lattice sizes and densities
* Density sweep experiments
* Measurement of macroscopic observables
* Visualization and post-processing tools

## Tech Stack

* C++
* OpenMP
* Python
* NumPy
* Matplotlib
* Jupyter

## Repository Structure

```text
src/        -> simulation code
analysis/   -> data analysis and plotting
notebooks/  -> exploratory notebooks
results/    -> generated outputs
```

## Running the Simulation

```bash
# compile
g++ main.cpp -O2 -o bml

# run
./bml
```

## Motivation

This project was developed to study collective behavior, non-equilibrium systems, and computational approaches to statistical physics.

## Future Improvements

* MPI/OpenMP scaling
* Better visualization pipeline
* Parameter sweep automation
* Performance benchmarking
