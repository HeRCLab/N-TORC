# Steps for Neural Network Architecture Search  
## Prepare training environment:  
Create a docker with tensorflow-gpu and install below libraries:  
numpy, scikit-learn, tensorflow, plotly, optuna  
## For training Neural Networks:  
Inside **dropbear_v45Data_optuna_new_BoTorchSampler.py**:  
step 1: Update data path of variables **pin_data_folderpath** and **acc_data_folderpath** with location of acceleration signals and pin location data.  
step 2: Update output directory in variables **out_file_tag**, **result_path**, and **rmse_cost_path**  
step 3: Run the file.  
