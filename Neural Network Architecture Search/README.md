# Steps for Neural Network Architecture Search  
## Prepare training environment:  
Create a docker with tensorflow-gpu and install below libraries:  
numpy, scikit-learn, tensorflow, plotly, optuna  
Commit changes to the docker  
Run all Python files inside the docker  

## For training Neural Networks:  
Run the file **dropbear_v45Data_Classification_optuna_new_BoTorchSampler.py**  

## Plot Pareto Front Diagram:  
Run the file **main_optuna_gen_pareto_front_new.py** 

step 1: For the **zoomed-in** pareto-front plot, set **MODEL_COST_LIMIT = 0.055** and for the **zoomed-out** version, set **MODEL_COST_LIMIT = 0.7**  

## Generate Results for Model Comparison:  
Run the file **dropbear_v45Data_Classification_optuna_new_BoTorchSampler_model_comparison.py**  

step 1: Set **IF_RUNNING_FOR_MORE_ACCURATE_MODEL** to True and False for results of more and less accurate models, respectively, and run the file once with setting **IF_RUNNING_FOR_MORE_ACCURATE_MODEL** to True and then run again by setting this variable to False.  

## Prepare Data for One Hump (Fig. 7 of paper) result plot:  
Run the file **data_processing_for_one_hump.py**  

## Plot One Hump (Fig. 7 of paper) result plot: 
Inside the folder one_hump_outputs, run the file **one_hump_plot_code.m**  
