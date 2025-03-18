# Prepare training environment:  
&nbsp;&nbsp;&nbsp;&nbsp;Create a docker with tensorflow-gpu and install below libraries:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;numpy, scikit-learn, tensorflow, plotly, optuna  
# For training Neural Networks:  
&nbsp;&nbsp;&nbsp;&nbsp;Inside **dropbear_v45Data_optuna_new_BoTorchSampler.py**:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step 1: Update data path of variables **pin_data_folderpath** and **acc_data_folderpath** with location of acceleration signals and pin location data.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step 2: Update output directory in variables **out_file_tag**, **result_path**, and **rmse_cost_path**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step 3: Run the file.  
