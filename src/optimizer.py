import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from gurobipy import Model, GRB, quicksum
import os
import yaml
from sklearn.tree import export_graphviz
#import graphviz
import warnings
import matplotlib.pyplot as plt
import math
import sys
import csv
import time
warnings.filterwarnings("ignore")
start_time=time.time()
# Function to validate the reuse factor
def validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = np.ceil((n_in * n_out) / multfactor)
    is_valid = (multiplier_limit % n_out == 0) or (rf >= n_in)
    is_valid = is_valid and ((rf % n_in == 0) or (rf < n_in))
    is_valid = is_valid and ((n_in * n_out) % rf == 0)
    return is_valid

# Function to get valid reuse factors
def get_valid_reuse_factors(n_in, n_out):
    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        if validate_reuse_factor(n_in, n_out, rf):
            valid_reuse_factors.append(rf)
    return valid_reuse_factors
class ModelAnalyzer:
    def __init__(self):
        self.first_conv1d = True 

    def get_layer_mult_size(self, layer):
        if 'Dense' in layer.class_name:
            n_in = layer.get_attr('n_in') 
            n_out = layer.get_attr('n_out')
            return n_in, n_out
        if 'Conv1D' in layer.class_name:
            if self.first_conv1d:
                n_in = layer.get_attr('hls_in') * layer.get_attr('filt_width') 
                self.first_conv1d = False  
            else:
                n_in = (layer.get_attr('n_chan') or 1) * (layer.get_attr('filt_width') or 1)
            n_out = layer.get_attr('n_filt') or 1
            print(f"n_out:{n_out}")
            return n_in, n_out
       

        if 'LSTM' in layer.class_name:
            n_in = layer.get_attr('n_in')
            print(f"lstm:{n_in}")
            n_out = layer.get_attr('n_out')  * 4
            print(f"lstm_out:{n_out}")# 4 gates in LSTM
            n_in_recr = layer.get_attr('n_out')
            print(f"lstm_in_recr:{n_in_recr}")
            n_out_recr = n_out
            print(f"lstm_out_rece:{n_out_recr}")
            return n_in, n_out, n_in_recr, n_out_recr

        raise Exception(f'Cannot get mult size for layer {layer.name} ({layer.class_name})')
    def div_roundup(self, a, b):
     
        return math.ceil(a / b)

    def get_layer_block_factor(self, n_in, n_out, reuse_factor):
        block_factor = self.div_roundup(n_in * n_out, reuse_factor)
        return block_factor
   
    def analyze_model(self, layers):
        valid_factors = {}
        block_factors = {}
        for layer in layers:
            if 'LSTM' in layer.class_name:
                n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)

                if n_in != n_in_recr or n_out != n_out_recr:
                   
                  
                   reuse_factors = get_valid_reuse_factors(n_in_recr, n_out_recr)
                else:
                    reuse_factors = get_valid_reuse_factors(n_in, n_out)
            else:
                n_in, n_out = self.get_layer_mult_size(layer)
                reuse_factors = get_valid_reuse_factors(n_in, n_out)
            if n_in == 0 or n_out == 0:
               print(f"Error: n_in or n_out is None for layer {layer.name} ({layer.class_name})")
               continue
            valid_factors[layer.name] = reuse_factors
            block_factors[layer.name] = {
              rf: self.get_layer_block_factor(n_in, n_out, rf) for rf in reuse_factors
            }
            print(f"Valid reuse factors for layer {layer.name} ({layer.class_name}): {reuse_factors}")
            print(f"Block factors for layer {layer.name} ({layer.class_name}): {block_factors[layer.name]}")
        return valid_factors, block_factors

    
   
class Layer:
    def __init__(self, class_name, config, build_config, model_config, is_first_cnn=False):
        self.class_name = class_name
        self.config = config
        self.build_config = build_config
        self.model_config = model_config 
        self.name = config.get('name', 'unknown')
        self.is_first_cnn = is_first_cnn  # Use the flag passed during initialization
    
    def get_attr(self, attr_name):
        if attr_name == 'n_in':
            input_shape = self.build_config.get('input_shape', [])
            if input_shape and len(input_shape) > 0:
                return input_shape[-1]
        if attr_name == 'n_out':
            return self.config.get('units', 1)
        if attr_name == 'n_chan' or attr_name == 'cnn_filters':
            return self.config.get('filters', 1)
        if attr_name == 'cnn_n_in':
            if 'Conv1D' in self.class_name:
                input_shape = self.build_config.get('input_shape', [])
                return input_shape[2] if len(input_shape) > 2 else None
        if attr_name == 'dense_size':
            if 'Dense' in self.class_name:
                return self.config.get('units')
        if attr_name == 'lstm_n_in':
            if 'LSTM' in self.class_name:
                input_shape = self.build_config.get('input_shape', [])
                return input_shape[2] if len(input_shape) > 2 else None
        if attr_name == 'filt_width':
            return self.config.get('kernel_size', [1])[0]
        if attr_name == 'dense_in':
            if 'Dense' in self.class_name:
                input_shape = self.build_config.get('input_shape', [])
                return input_shape[1]
        if attr_name == 'sequence_length':
            if 'Conv1D' in self.class_name:
                input_shape = self.build_config.get('input_shape', [])
                return input_shape[1] if len(input_shape) > 1 else None
            if 'LSTM' in self.class_name:
                input_shape = self.build_config.get('input_shape', [])
                return input_shape[1] if len(input_shape) > 1 else None
            return self.build_config.get('sequence_length', None)
        if attr_name == 'lstm_size':
            if 'LSTM' in self.class_name:
                return self.config.get('units')
        if attr_name == 'n_filt':
            return self.config.get('filters', 1)
        if attr_name == 'hls_in':
           input_shape=self.build_config.get('input_shape')
           return input_shape[2]

        return None


def train_and_evaluate(X, y, max_depth=None):
    # Split data into 80% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor(max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)

    # Predicting on the validation data
    y_val_pred = model.predict(X_val)

    # Calculating metrics for validation data
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mape_val, rmse_percentage_val = calculate_mape_rmse_percentage(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    # Print metrics for validation
    print(f"Validation Metrics: R² score: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%, RMSE %: {rmse_percentage_val:.2f}%")

    return model, r2_val, mae_val, rmse_val, mape_val, rmse_percentage_val
def calculate_mape_rmse_percentage(y_true, y_pred, epsilon=1e-10):
    # Filter out very small values from y_true
    threshold = 1e-3
    mask = y_true > threshold
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Calculate MAPE and RMSE percentage using filtered values
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon))) * 100
    rmse_percentage = (np.sqrt(np.mean((y_true_filtered - y_pred_filtered) ** 2)) / (np.mean(y_true_filtered) + epsilon)) * 100
    return mape, rmse_percentage





# Updated function to train layers separately and store additional metrics
def train_layers_separately(resources_file, latency_file, resource_columns, latency_columns, target_columns):
    # Load and process data
    resources_df = pd.read_csv(resources_file)
    latency_df = pd.read_csv(latency_file)

    # Initialize models dictionary
    rf_models = {}

    # Lists to store RMSE and MAE values
    rmse_values = []
    mae_values = []

    # Train models for resource targets
    print(f"\nModel Performance Metrics for Resources in {resources_file}:")
    for target in target_columns:
        if target in resources_df.columns:
            X = resources_df[resource_columns]
            y = resources_df[target]
            model, r2_val, mae_val, rmse_val, mape_val, rmse_percentage_val = train_and_evaluate(X, y)
            rf_models[f'resource_{target}'] = model
            rmse_values.append(rmse_val)  # Store the RMSE value for validation
            mae_values.append(mae_val)    # Store the MAE value for validation

            # Print additional metrics for validation
            print(f"Resource {target} - Validation R²: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%, RMSE %: {rmse_percentage_val:.2f}%")

    # Train models for latency targets
    print(f"\nModel Performance Metrics for Latency in {latency_file}:")
    for target in target_columns:
        if target in latency_df.columns:
            X = latency_df[latency_columns]
            y = latency_df[target]
            model, r2_val, mae_val, rmse_val, mape_val, rmse_percentage_val = train_and_evaluate(X, y)
            rf_models[f'latency_{target}'] = model
            rmse_values.append(rmse_val)  # Store the RMSE value for validation
            mae_values.append(mae_val)    # Store the MAE value for validation

            # Print additional metrics for validation
            print(f"Latency {target} - Validation R²: {r2_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%, RMSE %: {rmse_percentage_val:.2f}%")

    return rf_models












    # Columns to use from each dataset
resource_columns_conv = ['correct_reuse_factor_resource', 'n_inputs_resources', 'cnn_filters_resources', 'sequence_length_resource']
latency_columns_conv = ['correct_reuse_factor_latency', 'n_inputs_latency', 'cnn_filters_latency', 'sequence_length_latency']
resource_columns_lstm = ['correct_reuse_factor_lstm_resource', 'n_inputs_resources', 'sequence_length_resource', 'lstm_size']
latency_columns_lstm = ['correct_reuse_factor_lstm_latency', 'n_inputs_latency', 'sequence_length_latency', 'lstm_size']
resource_columns_dense = ['correct_reuse_factor_dense_resource', 'n_inputs_resources', 'sequence_length_resource', 'dense_size_resource']
latency_columns_dense = ['correct_reuse_factor_dense_latency', 'n_inputs_latency', 'sequence_length_latency', 'dense_size_latency']
# Target columns for the models
target_columns = ['latency_min', 'latency_max', 'bram_18k', 'lut', 'ff', 'dsp48e']
# Train models for Conv1D layers
rf_models_conv = train_layers_separately(
    'training_data_optimizer/collapsed_conv_resources.csv',
    'training_data_optimizer/conv_latency_collapsed.csv',
    resource_columns_conv,
    latency_columns_conv,
    target_columns
)
print("Keys in rf_models_conv:", rf_models_conv.keys())

# Train models for LSTM layers
rf_models_lstm = train_layers_separately(
    'training_data_optimizer/lstm_resources_collapsed.csv',
    'training_data_optimizer/lstm_latency_collapsed.csv',
    resource_columns_lstm,
    latency_columns_lstm,
    target_columns
)
# Train models for Dense layers
rf_models_dense = train_layers_separately(
    'training_data_optimizer/dense_resources_collapsed.csv',
    'training_data_optimizer/dense_latency_collapsed.csv',
    resource_columns_dense,
    latency_columns_dense,
    target_columns
)





# Extract layers and get valid reuse factors
analyzer = ModelAnalyzer()
layers = []
# Check if the filename is passed as an argument
if len(sys.argv) < 2:
    print("Usage: python optimizer_update.py <filename>")
    sys.exit(1)

# Load JSON data for the network
filename = sys.argv[1]
with open(filename, 'r') as f:
    model_config = json.load(f)

# Extract information from the model configuration
n_lstm = sum(1 for layer_config in model_config['config']['layers'] if 'LSTM' in layer_config['class_name'])
n_cnn = sum(1 for layer_config in model_config['config']['layers'] if 'Conv1D' in layer_config['class_name'])
cnn_filters = next((layer_config['config']['filters'] for layer_config in model_config['config']['layers'] if 'Conv1D' in layer_config['class_name']), 1)

# Extracting input length from InputLayer configuration
inputs = next((layer_config['config']['batch_input_shape'][1]
               for layer_config in model_config['config']['layers'] if layer_config['class_name'] == 'InputLayer'), 1)

# Parse the model configuration and create Layer instances
is_first_cnn = True  # Initialize flag for identifying the first Conv1D layer
total_layers = len(model_config['config']['layers'])

for i, layer_config in enumerate(model_config['config']['layers']):
    class_name = layer_config['class_name']
    config = layer_config['config']
    build_config = layer_config.get('build_config', {})
    #if class_name == 'Dense':
        #sequence_length = calculate_dense_sequence_length(i, n_lstm, n_cnn, cnn_filters, cnn_filters, inputs, total_layers)
        #build_config['sequence_length'] = sequence_length
    if class_name == 'Conv1D':
        if is_first_cnn:
            layer = Layer(class_name, config, build_config, model_config, is_first_cnn=True)
            is_first_cnn = False  # Switch off the flag after processing the first Conv1D layer
        else:
            layer = Layer(class_name, config, build_config, model_config, is_first_cnn=False)
    elif class_name in ['Dense', 'LSTM']:
        layer = Layer(class_name, config, build_config, model_config)
    if class_name in ['Dense', 'Conv1D', 'LSTM']:
        layers.append(layer)

# Analyze model to get valid reuse factors
valid_factors,block_factors = analyzer.analyze_model(layers)
def filter_valid_reuse_factors_with_block(layers, valid_factors, block_factors):
    filtered_factors = {}
    for layer_name, reuse_factors in valid_factors.items():
        layer = next(l for l in layers if l.name == layer_name)

        # Only keep reuse factors where the block factor is <= 4096
        filtered_factors[layer_name] = [
            rf for rf in reuse_factors if block_factors[layer_name][rf] <= 2048
        ]

        # Check if no reuse factors are valid for this layer
        if not filtered_factors[layer_name]:
            raise Exception(f"No valid reuse factors for layer {layer_name} within block factor constraints.")

        print(f"Valid reuse factors for layer {layer_name}: {filtered_factors[layer_name]}")
    
    return filtered_factors
filtered_factors = filter_valid_reuse_factors_with_block(layers, valid_factors, block_factors)


# Optimizer function

def optimize_reuse_factors_network(layers, filtered_factors, rf_models_conv, rf_models_lstm, rf_models_dense):
    model = Model('NetworkReuseFactorOptimization')
    model.setParam('OutputFlag', True)  # Enable solver output
    reuse_factors = {}

    # Initialization
    total_latency_min = 0
    total_latency_max = 0
    total_bram_usage = 0
    total_dsp_usage = 0
    total_ff_usage = 0
    total_lut_usage = 0

    # Iterate over each layer
    for layer_name, layer_factors in filtered_factors.items():
        layer = next(l for l in layers if l.name == layer_name)
        
        # Create binary variables for each possible reuse factor
        rf_binary_vars = {rf: model.addVar(vtype=GRB.BINARY, name=f"rf_{layer_name}_{rf}") for rf in layer_factors}
        
        # Create a reuse factor variable
        reuse_factor = model.addVar(vtype=GRB.INTEGER, name=f"reuse_factor_{layer_name}")
        
        # Constraint to ensure exactly one reuse factor is chosen
        model.addConstr(quicksum(rf_binary_vars[rf] for rf in layer_factors) == 1)
        
        # Linking the integer reuse factor variable with the binary variables
        model.addConstr(reuse_factor == quicksum(rf * rf_binary_vars[rf] for rf in layer_factors))
        
        # Extract data for prediction
        if 'Conv1D' in layer.class_name:
            n_in = layer.get_attr('cnn_n_in') or 0
            cnn_filters = layer.get_attr('cnn_filters') or 0
            sequence_length = layer.get_attr('sequence_length') or 0

            X_pred = np.array([[rf, n_in, cnn_filters, sequence_length] for rf in layer_factors])
        
            latency_min_pred = quicksum(
                rf_models_conv['latency_latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            
            latency_max_pred = quicksum(
                rf_models_conv['latency_latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_conv['resource_bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_conv['resource_dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_conv['resource_ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_conv['resource_lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        elif 'LSTM' in layer.class_name:
            n_in = layer.get_attr('lstm_n_in') 
            sequence_length = layer.get_attr('sequence_length') 
            lstm_size = layer.get_attr('lstm_size') 

            X_pred = np.array([[rf, n_in, sequence_length, lstm_size] for rf in layer_factors])

            latency_min_pred = quicksum(
                rf_models_lstm['latency_latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            latency_max_pred = quicksum(
                rf_models_lstm['latency_latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_lstm['resource_bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_lstm['resource_dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_lstm['resource_ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_lstm['resource_lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        elif 'Dense' in layer.class_name:
            n_in = layer.get_attr('dense_in') 
            dense_size = layer.get_attr('dense_size') or 1

            X_pred = np.array([[rf, n_in, 1, dense_size] for rf in layer_factors])

            latency_min_pred = quicksum(
                rf_models_dense['latency_latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            latency_max_pred = quicksum(
                rf_models_dense['latency_latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_dense['resource_bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_dense['resource_dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_dense['resource_ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_dense['resource_lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        # Aggregate latency and resource usage
        total_latency_min += latency_min_pred
        total_latency_max += latency_max_pred
        total_bram_usage += bram_usage_pred
        total_dsp_usage += dsp_usage_pred
        total_ff_usage += ff_usage_pred
        total_lut_usage += lut_usage_pred

    
     #Latency constraint
    model.addConstr(total_latency_min <= 50000, "Latency_constraint")
    # Constraints
    model.addConstr(total_bram_usage <= 624  + slack_bram, "BRAM_constraint")
    model.addConstr(total_dsp_usage <= 1728  + slack_dsp, "DSP_constraint")
    model.addConstr(total_ff_usage <= 460800  + slack_ff, "FF_constraint")
    model.addConstr(total_lut_usage <= 230400  + slack_lut, "LUT_constraint")

    # Objective function
    model.setObjective(total_bram_usage + total_dsp_usage + total_ff_usage + total_lut_usage ,GRB.MINIMIZE)
    # Optimize the model
    model.optimize()

    # Warning if slack being used
    slack_used = any(var.X > 0 for var in [slack_bram, slack_dsp, slack_ff, slack_lut])
    if slack_used:
        print("WARNING: Slack variables were used, indicating that the network may not fit onto the ZCU104 Board.")
    
    # Extract optimal reuse factors 
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        print(f"\nTotal predicted_latency_max: {total_latency_max.getValue():.2f}")
        print(f"Total predicted_lut: {total_lut_usage.getValue():.2f}")
        print(f"Total predicted_bram: {total_bram_usage.getValue():.2f}")
        print(f"Total predicted_dsp: {total_dsp_usage.getValue():.2f}")
        print(f"Total predicted_ff: {total_ff_usage.getValue():.2f}\n")

        for layer_name in filtered_factors.keys():
            selected_rf = [rf for rf in filtered_factors[layer_name] if model.getVarByName(f"rf_{layer_name}_{rf}").X > 0.5]

            if selected_rf:
                reuse_factors[layer_name] = selected_rf[0]
                rf = selected_rf[0]
                layer = next((layer for layer in layers if layer.name == layer_name), None)
                if layer and 'Conv1D' in layer.class_name:
                    # Construct X_pred for the selected reuse factor
                    n_in = layer.get_attr('cnn_n_in') or 0
                    cnn_filters = layer.get_attr('cnn_filters') or 0
                    sequence_length = layer.get_attr('sequence_length') or 0
                    X_pred = np.array([[rf, n_in, cnn_filters, sequence_length]])

                    # Perform the prediction
                    predicted_lut_value = rf_models_conv['resource_lut'].predict(X_pred)[0]
                    print(f"Layer: {layer.name}, Reuse Factor: {rf}, Predicted LUT: {predicted_lut_value}")
            else:
                print(f"No feasible solution found for layer {layer_name} that fits the resource constraints.")
    else:
        print(f"Optimizer did not find an optimal solution. Status: {model.status}")
        
    return reuse_factors







# Count the number of valid reuse factors for each layer
valid_counts = {layer_name: len(factors) for layer_name, factors in filtered_factors.items()}

# Calculate the total number of possible combinations
total_combinations = 1
for count in valid_counts.values():
    total_combinations *= count

print("Number of valid reuse factors for each layer:")
for layer_name, count in valid_counts.items():
    print(f"{layer_name}: {count}")
print(f"\nTotal number of possible combinations: {total_combinations/1000000} million")

# Optimize reuse factors for the entire network
#output_csv_path = 'network_predictions_total.csv'
start_time=time.time()
optimal_reuse_factors = optimize_reuse_factors_network(layers, filtered_factors, rf_models_conv, rf_models_lstm, rf_models_dense)

print("Optimal Reuse Factors for Network:")
for layer_name, reuse_factor in optimal_reuse_factors.items():
    print(f"{layer_name}: {reuse_factor}")




def save_predictions_to_csv(layers, filtered_factors, rf_models_conv, rf_models_lstm, rf_models_dense, output_csv_paths):
    # List to store data for each layer type
    data_conv1d = []
    data_lstm = []
    data_dense = []

    # Iterate over each layer and its valid reuse factors
    for layer_name, layer_factors in filtered_factors.items():
        layer = next(l for l in layers if l.name == layer_name)

        for i, rf in enumerate(layer_factors):
            if 'Conv1D' in layer.class_name:
                # Extract attributes
                n_in = layer.get_attr('cnn_n_in') or 0
                cnn_filters = layer.get_attr('cnn_filters') or 0
                sequence_length = layer.get_attr('sequence_length') or 0

                
                X_pred = np.array([[rf, n_in, cnn_filters, sequence_length]])

                predicted_lut = rf_models_conv['resource_lut'].predict(X_pred)[0]
                predicted_ff = rf_models_conv['resource_ff'].predict(X_pred)[0]
                predicted_bram = rf_models_conv['resource_bram_18k'].predict(X_pred)[0]
                predicted_dsp = rf_models_conv['resource_dsp48e'].predict(X_pred)[0]
                predicted_latency_min = rf_models_conv['latency_latency_min'].predict(X_pred)[0]
                predicted_latency_max = rf_models_conv['latency_latency_max'].predict(X_pred)[0]

                # Append the data to the list for Conv1D layers
                data_conv1d.append({
                    'layer_name': layer_name,
                    'reuse_factor': rf,
                    'sequence_length': sequence_length,
                    'cnn_filters': cnn_filters,
                    'cnn_in': n_in,
                    'predicted_lut': predicted_lut,
                    'predicted_ff': predicted_ff,
                    'predicted_bram': predicted_bram,
                    'predicted_dsp': predicted_dsp,
                    'predicted_latency_min': predicted_latency_min,
                    'predicted_latency_max': predicted_latency_max
                })

            elif 'LSTM' in layer.class_name:
                # Extract attributes
                n_in = layer.get_attr('lstm_n_in') 
                lstm_size = layer.get_attr('lstm_size') 
                sequence_length = layer.get_attr('sequence_length') 

               
                X_pred = np.array([[rf, n_in, sequence_length, lstm_size]])

                predicted_lut = rf_models_lstm['resource_lut'].predict(X_pred)[0]
                predicted_ff = rf_models_lstm['resource_ff'].predict(X_pred)[0]
                predicted_bram = rf_models_lstm['resource_bram_18k'].predict(X_pred)[0]
                predicted_dsp = rf_models_lstm['resource_dsp48e'].predict(X_pred)[0]
                predicted_latency_min = rf_models_lstm['latency_latency_min'].predict(X_pred)[0]
                predicted_latency_max = rf_models_lstm['latency_latency_max'].predict(X_pred)[0]

                
                data_lstm.append({
                    'layer_name': layer_name,
                    'reuse_factor': rf,
                    'sequence_length': sequence_length,
                    'lstm_size': lstm_size,
                    'n_in':n_in,
                    'predicted_lut': predicted_lut,
                    'predicted_ff': predicted_ff,
                    'predicted_bram': predicted_bram,
                    'predicted_dsp': predicted_dsp,
                    'predicted_latency_min': predicted_latency_min,
                    'predicted_latency_max': predicted_latency_max
                })

            elif 'Dense' in layer.class_name:
                # Extract attributes
                n_in = layer.get_attr('dense_in') 
                dense_size = layer.get_attr('dense_size') 

               
                X_pred = np.array([[rf, n_in, 1, dense_size]])

                predicted_lut = rf_models_dense['resource_lut'].predict(X_pred)[0]
                predicted_ff = rf_models_dense['resource_ff'].predict(X_pred)[0]
                predicted_bram = rf_models_dense['resource_bram_18k'].predict(X_pred)[0]
                predicted_dsp = rf_models_dense['resource_dsp48e'].predict(X_pred)[0]
                predicted_latency_min = rf_models_dense['latency_latency_min'].predict(X_pred)[0]
                predicted_latency_max = rf_models_dense['latency_latency_max'].predict(X_pred)[0]

                
                data_dense.append({
                    'layer_name': layer_name,
                    'reuse_factor': rf,
                    'dense_size': dense_size,
                    'n_in':n_in,
                    'sequence_length': 1,
                    'predicted_lut': predicted_lut,
                    'predicted_ff': predicted_ff,
                    'predicted_bram': predicted_bram,
                    'predicted_dsp': predicted_dsp,
                    'predicted_latency_min': predicted_latency_min,
                    'predicted_latency_max': predicted_latency_max
                })

    
    pd.DataFrame(data_conv1d).to_csv(output_csv_paths['conv'], index=False)
    pd.DataFrame(data_lstm).to_csv(output_csv_paths['lstm'], index=False)
    pd.DataFrame(data_dense).to_csv(output_csv_paths['dense'], index=False)

    print(f"Predictions saved to {output_csv_paths['conv']}, {output_csv_paths['lstm']}, {output_csv_paths['dense']}")


output_csv_paths = {
    'conv': 'conv1d_predictions.csv',
    'lstm': 'lstm_predictions.csv',
    'dense': 'dense_predictions.csv'
}


save_predictions_to_csv(layers, filtered_factors, rf_models_conv, rf_models_lstm, rf_models_dense, output_csv_paths)
end_time=time.time()
elapsed_time=end_time-start_time
print(f"\n Process completed in {elapsed_time:.2f} seconds")

# Function to generate YAML for HLS4ML
def generate_yaml_for_hls4ml(json_path, filename, optimized_reuse_factors):
    # Construct paths for JSON and H5 files
    model_json_path = json_path
    model_h5_path = os.path.splitext(json_path)[0] + ".h5"

    # Load the Keras model JSON
    try:
        with open(model_json_path, "r") as json_file:
            model_json = json.load(json_file)
    except (IOError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not read or parse {model_json_path}: {e}")
        return

    # Initialize LayerName section
    layer_name_config = {}
    unique_layer_names = []

    # Extract layer names and assign optimized reuse factors
    for layer in model_json["config"]["layers"]:
        layer_class = layer["class_name"]
        if layer_class in ["Conv1D", "LSTM", "Dense"]:
            layer_name = layer["config"].get("name", "")
            if layer_name and layer_name not in unique_layer_names:
                unique_layer_names.append(layer_name)

    # Assign optimized reuse factors to each layer
    for layer_name in unique_layer_names:
        reuse_factor = optimized_reuse_factors.get(layer_name, optimized_reuse_factors.get(unique_layer_names[0], 16))
        layer_name_config[layer_name] = {
            "ReuseFactor": reuse_factor,
            "Strategy": "Resource"
        }

    # Find the highest reuse factor in layer_name_config
    highest_reuse_factor = max(config["ReuseFactor"] for config in layer_name_config.values())

    # Build YAML content
    yaml_content = {
        "Backend": "Vivado",
        "Part": "xczu7ev-ffvc1156-2-e",
        "ClockPeriod": 4,
        "IOType": "io_stream",
        "KerasJson": model_json_path,
        "KerasH5": model_h5_path,
        "OutputDir": os.path.splitext(filename)[0],
        "ProjectName": os.path.splitext(filename)[0],
        "HLSConfig": {
            "Model": {
                "Precision": "ap_fixed<16,8,AP_RND,AP_SAT>",
                "ReuseFactor": highest_reuse_factor,
                "Strategy": "Resource",
            },
            "LayerName": layer_name_config
        }
    }

    # Save the YAML content to a file in the current directory
    yaml_path = os.path.join(os.getcwd(), os.path.splitext(filename)[0] + ".yaml")

    try:
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)
        print(f"YAML file saved at: {yaml_path}")
    except IOError as e:
        print(f"ERROR: Could not open {yaml_path} for writing: {e}")


generate_yaml_for_hls4ml(filename, filename, optimal_reuse_factors)
