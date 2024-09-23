# HLS4ML explorer

Usage: python src/optmizer.py <json_file>


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

graphviz: Graph generation(optional).

HLS4ML: For hardware synthesis after YAML generation.

yaml: YAML file handling.

json: JSON file handling.

os: Operating system interaction.

subprocess: Running shell commands.

multiprocessing: Parallel processing.



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

Purpose of the Slack Penalty:
Penalty for Constraint Violations:

The slack variables (slack BRAM,slack DSP,slack FF,slackLUT ) represent the amount by which the resource constraints are violated. In an ideal solution, no slack is used, meaning the network fits within the resource limits.
By multiplying the sum of the slack variables by a large constant , the optimization problem penalizes solutions that require more resources than available. This forces the optimizer to prioritize solutions that minimize slack usage, i.e., it discourages exceeding hardware resource limits.
Balancing Latency and Resource Usage:
The primary objective is to minimize latency. However, without this penalty, the optimizer might find a solution that minimizes latency but exceeds resource limits.
By adding a large penalty for slack, the optimizer is encouraged to find a trade-off between minimizing latency and fitting the design within the available hardware resources. The weight of 1000 ensures that exceeding resource limits is heavily penalized, making it undesirable unless absolutely necessary.
Flexibility in Resource Allocation:
Slack variables provide flexibility to allow some violations if it's not possible to meet resource constraints strictly. However, the large multiplier ensures that this happens only as a last resort.


