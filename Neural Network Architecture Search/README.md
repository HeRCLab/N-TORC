# Steps for Neural Network Architecture Search  
## Prepare training environment:  
Create a docker with tensorflow-gpu and install below libraries:  
numpy, scikit-learn, tensorflow, plotly, optuna  
Commit changes to the docker  
Run all Python files inside the docker  

## For training Neural Networks:  
Inside **dropbear_v45Data_Classification_optuna_new_BoTorchSampler.py**:  
  
step 1: Update data path of variables **pin_data_folderpath** and **acc_data_folderpath** with location of acceleration signals and pin location data.  
  
step 2: Update output directory in variables **out_file_tag**, **result_path**, and **rmse_cost_path**  
  
step 3: Run the file.  


## Plot Pareto Front Diagram:  
Inside **main_optuna_gen_pareto_front_new.py**:  

step 1: Update variable **dir_path** with the path of the results' directory used in the training step.  

step 2: Update variable **FILEPATH** with the output directory to store the generated plots  

step 3: For the **zoomed-in** pareto-front plot, set **MODEL_COST_LIMIT = 0.055** and for the **zoomed-out** version, set **MODEL_COST_LIMIT = 0.7**  

step 4: Run the file. 

## Generate Results for Model Comparison:  
Inside **dropbear_v45Data_Classification_optuna_new_BoTorchSampler_model_comparison.py**:  

step 1: Update data path of variables **pin_data_folderpath** and **acc_data_folderpath** with location of acceleration signals and pin location data.  

step 2: Set **IF_RUNNING_FOR_MORE_ACCURATE_MODEL** to True and False for results of more and less accurate models respectively.  
  
step 3: Update the output directory in variables **out_file_tag**, **result_path**, **rmse_cost_path**, and **acc_pin_ref_pred_filepath** appropriately  

step 4: Run the file once with setting **IF_RUNNING_FOR_MORE_ACCURATE_MODEL** to True and then run again by setting this variable to False.  
  
step 5: Select one of the test datasets from *Standard_Index_Set dataset* i.e. Standard_Index_Set_test5, or Standard_Index_Set_test16, or Standard_Index_Set_test10 for generating the plot, and select the required data files for the next step from the data files saved in this step.  

## Prepare Data for One Hump (Fig. 7 of paper) result plot:  
Inside **data_processing_for_one_hump.py**:  

step 1: Update **dir_path** with the directory path containing the selected input acceleration signal data file, corresponding reference pin location file, and corresponding 2 predicted pin location files, one for a more accurate and another for less accurate model, and time values data file.  
  
step 2: Specify those filenames in variables **filepath_acc, filepath_pin, filepath_pred_pin_bad, filepath_pred_pin_good, and filepath_time**.  
  
step 3: Update **out_dir** with the output directory path to contain intermediate output files.  

step 4: Run the file.  

## Plot One Hump (Fig. 7 of paper) result plot: 
Inside **one_hump_plot_code.m**:  

step 1: Set the appropriate path for variables, path to time values to *x_filename*, acceleration signal data from the above step to *y0_filename*, reference pin locations data from the above step to *y_filename*, predicted pin locations data from more and less accurate models from the above step to *y2_filename* and *y3_filename* respectively.  

step 2: Run the file in Matlab.


