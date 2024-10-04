# HLS4ML explorer

Usage: 

conda env create -f environment.yml

conda activate hls4ml_explorer

python src/optmizer.py <json_file>


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




