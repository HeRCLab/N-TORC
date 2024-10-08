# HLS4ML explorer

Example Usage: 

conda env create -f environment.yml

conda activate hls4ml_explorer

python src/optmizer.py three_lstm_model.json


This will create a yaml file with the optimized reuse factor for each layer in the current directory.
Linux Users:
source Vivado/2019.1/settings.sh (vivado environment script)

convert the yaml files into hls code:
hls4ml convert -c three_lstm_model.yaml

Go to the directory three_lstm_model/
run vivado_hls -f build_prj.tcl

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




