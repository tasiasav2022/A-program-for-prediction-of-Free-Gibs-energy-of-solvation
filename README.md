# SolvPredict

A Python program for predicting the Gibbs free energy of solvation (Δ*G*°<sub>solv</sub>) for organic solvents.

## Description

**SolvPredict** uses a trained dense neural network to predict the Gibbs free energy of solvation in organic solvents.

**Authors:** Anastasia Savchenko and Sergei Vyboishchikov (Universitat de Girona, Spain)

**Date:** July 2026

## Requirements

The file `ModelData.pkl`, containing the model parameters, is required.

The following Python packages are required: `RDKit`, `NumPy`, `pickle`

## Usage

```SolvPredict.py solute_smiles1 solute_smiles2 ... solvent```

### Arguments

- `solute_smiles1`, `solute_smiles2`, ...: SMILES strings of one or more solutes
- `solvent`: name of the solvent

## How it works

The program:

1. Reads the SMILES strings of the solutes and the solvent name from the command line.
2. Converts the SMILES strings into count-based Morgan fingerprints using RDKit.
3. Applies the trained neural network.
4. Outputs the predicted Δ*G*°<sub>solv</sub> for each solute.
