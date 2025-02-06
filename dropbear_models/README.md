#Running the Optimizer Script

1. Activate the conda environment
```bash
conda activate hls4ml_explorer
```
2. Run the script
   ```bash
   python optimizer_run.py
   ```
   The above script runs all the optimized networks that came from the bayesian optimizer through the ILP optimizer which generates the yaml file and stores the predicted LUTs , DSPs and latency in a networks_predicted.csv file. 
