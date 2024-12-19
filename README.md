# HLS4ML explorer

Example Usage: 
## Setting up the Environment

To set up the environment using `conda`, you can use the provided `environment.yml` file. Follow these steps:

1. Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
   
2. Open a terminal (or Anaconda Prompt).

3. Run the following command to create the environment:

    ```bash
    conda env create -f environment.yml
    ```

4. Once the environment is created, activate it:

    ```bash
    conda activate hls4ml_explorer
    ```

5. Verify the environment is activated and all dependencies are installed:

    ```bash
    conda list
    ```

6. To run the optimizer with the provided JSON model, use the following command:

    ```bash
    python src/optimizer.py three_lstm_model.json
    ```

You are now ready to use the `hls4ml_explorer` environment and run the optimizer with the specified model.


This will create a yaml file with the optimized reuse factor for each layer in the current directory.
Linux Users:
source Vivado/2019.1/settings.sh (vivado environment script)

convert the yaml files into hls code:
  ```bash
hls4ml convert -c three_lstm_model.yaml
```
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


If you use this tool in your research, please cite the corresponding publication as follows:
 ```bash
\copyrightyear{2025}
\acmYear{2025}
\setcopyright{rightsretained}
\acmConference[FPGA '25]{Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays}{February 27-March 1, 2025}{Monterey, CA, USA}
\acmBooktitle{Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays (FPGA '25), February 27-March 1, 2025, Monterey, CA, USA} \acmDOI{10.1145/3706628.3708848} \acmISBN{979-8-4007-1396-5/25/02}
 ```

