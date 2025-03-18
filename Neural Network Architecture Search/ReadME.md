# Steps for Neural Network Architecture Search  
## Prepare training environment:  
Create a docker with tensorflow-gpu and install below libraries:  
numpy, scikit-learn, tensorflow, plotly, optuna  
## For training Neural Networks:  
Inside **dropbear_v45Data_optuna_new_BoTorchSampler.py**:  
  
step 1: Update data path of variables **pin_data_folderpath** and **acc_data_folderpath** with location of acceleration signals and pin location data.  
  
step 2: Update output directory in variables **out_file_tag**, **result_path**, and **rmse_cost_path**  
  
step 3: Run the file.  


## Plot Pareto Front Diagram:  
Inside **main_optuna_gen_pareto_front_new.py**:  

step 1: Update variable **dir_path** with the path of the results' directory used in training step.  

step 2: Update variable **FILEPATH** with the output directory to store the generated plots  

step 3: For the **zoomed-in** pareto-front plot, set **MODEL_COST_LIMIT = 0.055** and for the **zoomed-out** version, set **MODEL_COST_LIMIT = 0.7**  

step 4: Run the file. 

