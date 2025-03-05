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

Cite our work
```bash
@inproceedings{10.1145/3706628.3708848,
author = {Singh, Suyash Vardhan and Ahmad, Iftakhar and Andrews, David and Huang, Miaoqing and Downey, Austin R. J. and Bakos, Jason D.},
title = {Resource Scheduling for Real-Time Machine Learning},
year = {2025},
isbn = {9798400713965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3706628.3708848},
doi = {10.1145/3706628.3708848},
abstract = {Data-driven physics models offer the potential for substantially increasing the sample rate for applications in high-rate cyberphys- ical systems, such as model predictive control, structural health monitoring, and online smart sensing. Making this practical re- quires new model deployment tools that search for networks with maximum accuracy while meeting both real-time performance and resource constraints. Tools that generate customized architectures for machine learning models, such as HLS4ML and FINN, require manual control over latency and cost trade-offs for each layer. This poster describes a proposed end-to-end framework that combines Bayesian optimization for neural architecture search with Integer Linear Optimization of layer cost-latency trade-off using HLS4ML ''reuse factors''. The proposed framework is shown in Fig. 1 and consists of a performance model training phase and two model deployment stages. The performance model training phase generates training data and trains a model to predict the resource cost and latency of an HLS4ML deployment of a given layer and associated reuse factor on a given FPGA. The first model deployment stage takes training, test, and validation data for a physical system-in this case, the Dynamic Reproduction of Projectiles in Ballistic Environments for Advanced Research (DROPBEAR) dataset-and searches the hyper- parameter space for Pareto optimal models with respect to latency and workload, as measured by the number of multiplies required for one forward pass. For each of the models generated, a second stage uses the performance model to optimize the reuse factor of each layer to guarantee that the whole model meets the resource constraint while minimizing end-to-end latency. Table 1 shows the benefit of the reuse factor optimizer that comprises the second stage of the model deployment phase, The results compare the performance of a baseline stochastic search to that of our proposed optimizer for an example model consisting of four convolutional layers, three LSTM layers, and one dense layer. The results show sample stochastic search runs having 1K, 10K, 100K, and 1M trials over a total search space of 209 million reuse factor permutations. The stochastic search reaches a point of diminishing returns with latency 205 𝜂 while the optimizer achieves a latency of 190 𝜂 and requires roughly 1000X less search time.},
booktitle = {Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
pages = {50},
numpages = {1},
keywords = {hardware acceleration, high-level synthesis (hls), real-time control systems, resource scheduling},
location = {Monterey, CA, USA},
series = {FPGA '25}
}
```
