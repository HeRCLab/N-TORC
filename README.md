# hls4ml_explorer

Usage: python src/optmizer.py <json_file>

Reuse Factor Optimization Tool for Neural Network Layers
This tool performs reuse factor optimization for neural network layers (Conv1D, LSTM, and Dense) with a focus on minimizing resource usage and latency. It leverages RandomForestRegressor models for resource and latency prediction and optimizes reuse factors using Gurobi. The tool also generates YAML configuration files for HLS4ML, providing an efficient pipeline for hardware synthesis.

Features
Predicts LUT, BRAM, FF, DSP usage, and latency for Conv1D, LSTM, and Dense layers.
Optimizes reuse factors for each layer using Gurobi to minimize latency while respecting hardware resource constraints.
Generates HLS4ML-compatible YAML files for FPGA implementation.
Supports evaluation and training of RandomForest models for resource and latency predictions.
Prerequisites
Required Libraries:
pandas: Data manipulation and analysis.
numpy: Numerical computing.
scikit-learn: Machine learning algorithms (RandomForestRegressor).
gurobipy: Optimization with Gurobi.
matplotlib: Plotting (optional).
yaml: YAML file handling.
graphviz: Graph generation.
Optional Tools:
HLS4ML: For hardware synthesis after YAML generation.


Mathematical Model
The optimization tool minimizes latency while respecting hardware resource constraints (BRAM, DSP, FF, LUT). For a detailed breakdown of the objective function and constraints, refer to the Objective Function and Constraints section below.

Minimize: 
    Sum (i=1 to L) of (Latency_max^i) 
    + 1000 * (slack_BRAM + slack_DSP + slack_FF + slack_LUT)

BRAM Constraint: 
    Sum(BRAM usage) ≤ BRAM limit + slack_BRAM

DSP Constraint: 
    Sum(DSP usage) ≤ DSP limit + slack_DSP

FF Constraint: 
    Sum(FF usage) ≤ FF limit + slack_FF

LUT Constraint: 
    Sum(LUT usage) ≤ LUT limit + slack_LUT

