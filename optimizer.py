import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from gurobipy import Model, GRB, quicksum
import os
import yaml
import json
from sklearn.metrics import r2_score



#import warnings
#warnings.filterwarnings("ignore")

#Random forest regresor
def train_rf_model(X, y):
    #splitting the data: 80% training and 20% test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model =  RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    

    
    y_pred = model.predict(X_test)
    
    
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

#Load the data
resources_conv_df = pd.read_csv('collapsed_conv_resources.csv')
latency_conv_df = pd.read_csv('conv_latency_collapsed.csv')



# Rename columns in latency data to match resource data
latency_conv_df = latency_conv_df.rename(columns={
    'correct_reuse_factor_latency': 'correct_reuse_factor_resource',
    'n_inputs_latency': 'n_inputs_resources',
    'cnn_filters_latency': 'cnn_filters_resources',
    'sequence_length_latency': 'sequence_length_resource'
})

# Merge the dataframes on the common columns
merged_conv_data = pd.merge(resources_conv_df, latency_conv_df, on=[
    'correct_reuse_factor_resource', 'n_inputs_resources',
    'cnn_filters_resources', 'sequence_length_resource'
])

# Extract features and targets for Random Forest
X_conv = merged_conv_data[['correct_reuse_factor_resource', 'n_inputs_resources', 'cnn_filters_resources', 'sequence_length_resource']]
y_min_conv = merged_conv_data['latency_min']
y_max_conv = merged_conv_data['latency_max']
bram_conv = merged_conv_data['bram_18k']
lut_conv = merged_conv_data['lut']
ff_conv = merged_conv_data['ff']
dsp_conv = merged_conv_data['dsp48e']

# Train Random Forest models for Conv1D
rf_models_conv = {}
print("R² Scores for Conv1D Layers:")
for key, y in zip(['latency_min', 'latency_max', 'bram_18k', 'lut', 'ff', 'dsp48e'],
                  [y_min_conv, y_max_conv, bram_conv, lut_conv, ff_conv, dsp_conv]):
    model, r2 = train_rf_model(X_conv, y)
    rf_models_conv[key] = model
    print(f"R² score for Conv1D {key}: {r2:.4f}")




resources_lstm_df = pd.read_csv('lstm_resources_collapsed.csv')
latency_lstm_df = pd.read_csv('lstm_latency_collapsed.csv')



latency_lstm_df = latency_lstm_df.rename(columns={
    'correct_reuse_factor_lstm_latency': 'correct_reuse_factor_lstm_resource',
    'n_inputs_latency': 'n_inputs_resources',
    'sequence_length_latency': 'sequence_length_resource',
    'lstm_size': 'lstm_size'
})


merged_lstm_data = pd.merge(resources_lstm_df, latency_lstm_df, on=[
    'correct_reuse_factor_lstm_resource', 'n_inputs_resources',
    'sequence_length_resource', 'lstm_size'
])


X_lstm = merged_lstm_data[['correct_reuse_factor_lstm_resource', 'n_inputs_resources', 'sequence_length_resource', 'lstm_size']]
y_min_lstm = merged_lstm_data['latency_min']
y_max_lstm = merged_lstm_data['latency_max']
bram_lstm = merged_lstm_data['bram_18k']
lut_lstm = merged_lstm_data['lut']
ff_lstm = merged_lstm_data['ff']
dsp_lstm = merged_lstm_data['dsp48e']

# Train Random Forest models for LSTM
rf_models_lstm = {}
print("\nR² Scores for LSTM Layers:")
for key, y in zip(['latency_min', 'latency_max', 'bram_18k', 'lut', 'ff', 'dsp48e'],
                  [y_min_lstm, y_max_lstm, bram_lstm, lut_lstm, ff_lstm, dsp_lstm]):
    model, r2 = train_rf_model(X_lstm, y)
    rf_models_lstm[key] = model
    print(f"R² score for LSTM {key}: {r2:.4f}")

#Dense Layer data
resources_dense_df = pd.read_csv('dense_resources_collapsed.csv')
latency_dense_df = pd.read_csv('dense_latency_collapsed.csv')



latency_dense_df = latency_dense_df.rename(columns={
    'correct_reuse_factor_dense_latency': 'correct_reuse_factor_dense_resource',
    'n_inputs_latency': 'n_inputs_resources',
    'sequence_length_latency': 'sequence_length_resource',
    'dense_size_latency': 'dense_size_resource'
})


merged_dense_data = pd.merge(resources_dense_df, latency_dense_df, on=[
    'correct_reuse_factor_dense_resource', 'n_inputs_resources',
    'sequence_length_resource', 'dense_size_resource'
])


X_dense = merged_dense_data[['correct_reuse_factor_dense_resource', 'n_inputs_resources', 'sequence_length_resource', 'dense_size_resource']]
y_min_dense = merged_dense_data['latency_min']
y_max_dense = merged_dense_data['latency_max']
bram_dense = merged_dense_data['bram_18k']
lut_dense = merged_dense_data['lut']
ff_dense = merged_dense_data['ff']
dsp_dense = merged_dense_data['dsp48e']

#Train Random Forest model for Dense
rf_models_dense = {}
print("\nR² Scores for Dense Layers:")
for key, y in zip(['latency_min', 'latency_max', 'bram_18k', 'lut', 'ff', 'dsp48e'],
                  [y_min_dense, y_max_dense, bram_dense, lut_dense, ff_dense, dsp_dense]):
    model, r2 = train_rf_model(X_dense, y)
    rf_models_dense[key] = model
    print(f"R² score for Dense {key}: {r2:.4f}")

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
                n_in = layer.get_attr('filt_width')
                self.first_conv1d = False  
            else:
                n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
            return n_in, n_out

        if 'LSTM' in layer.class_name:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out') * 4  # 4 gates in LSTM
            n_in_recr = layer.get_attr('n_out')
            n_out_recr = n_out
            return n_in, n_out, n_in_recr, n_out_recr

        raise Exception(f'Cannot get mult size for layer {layer.name} ({layer.class_name})')

    def analyze_model(self, layers):
        valid_factors = {}
        for layer in layers:
            if 'LSTM' in layer.class_name:
                n_in, n_out, _, _ = self.get_layer_mult_size(layer)
            else:
                n_in, n_out = self.get_layer_mult_size(layer)

            # Add check to ensure n_in and n_out are not None
            if n_in is None or n_out is None:
                print(f"Error: n_in or n_out is None for layer {layer.name} ({layer.class_name})")
                continue

            reuse_factors = get_valid_reuse_factors(n_in, n_out)
            valid_factors[layer.name] = reuse_factors
            print(f"Valid reuse factors for layer {layer.name} ({layer.class_name}): {reuse_factors}")

        return valid_factors


class Layer:
    def __init__(self, class_name, config, build_config):
        self.class_name = class_name
        self.config = config
        self.build_config = build_config
        self.name = config.get('name', 'unknown')  

    def get_attr(self, attr_name):
        if attr_name == 'n_in':
            input_shape = self.build_config.get('input_shape', [])
            if input_shape and len(input_shape) > 0:
                return input_shape[-1]
        if attr_name == 'n_out':
            return self.config.get('units', 1)
        if attr_name == 'n_chan':
            return self.config.get('filters', 1)
        if attr_name == 'filt_width':
            return self.config.get('kernel_size', [1])[0]
        if attr_name == 'sequence_length':
            return self.build_config.get('sequence_length', None)  # Adjust as necessary
        if attr_name == 'n_filt':
            return self.config.get('filters', 1)
        return None
#Incorporated from our MATLAB code
def calculate_dense_sequence_length(layer_index, n_lstm, n_cnn, cnn_filters, lstm_size, inputs):
    if layer_index == 0:  # First Dense Layer
        if n_lstm > 0:  # LSTM layers present
            n_inputs = lstm_size * inputs / (2 ** n_cnn)
            sequence_length = inputs / (2 ** n_cnn)
        else:  # No LSTM layers
            n_inputs = cnn_filters * inputs / (2 ** n_cnn)
            sequence_length = inputs / (2 ** n_cnn)
    else:  # Subsequent Dense Layers
        n_inputs = None 
        sequence_length = inputs / (2 ** n_cnn)
    
    # Last Dense Layer
    if layer_index == len(layers) - 1:
        n_inputs = 1
        sequence_length = 1

    return sequence_length

#Optimizer
def optimize_reuse_factors_network(layers, valid_factors, rf_models_conv, rf_models_lstm, rf_models_dense):
    model = Model('NetworkReuseFactorOptimization')
    model.setParam('OutputFlag', True)  # turn off solver output
    reuse_factors = {}

    # Initialization
    total_latency_min = 0
    total_latency_max = 0
    total_bram_usage = 0
    total_dsp_usage = 0
    total_ff_usage = 0
    total_lut_usage = 0

    # Iterate over each layer
    for layer_name, layer_factors in valid_factors.items():
        layer = next(l for l in layers if l.name == layer_name)

        # Created binary variables for each possible reuse factor
        rf_binary_vars = {rf: model.addVar(vtype=GRB.BINARY, name=f"rf_{layer_name}_{rf}") for rf in layer_factors}
        
        # Created a reuse factor variable
        reuse_factor = model.addVar(vtype=GRB.INTEGER, name=f"reuse_factor_{layer_name}")
        
        # Constraint to ensure exactly one reuse factor is chosen
        model.addConstr(quicksum(rf_binary_vars[rf] for rf in layer_factors) == 1)
        
        # Linking the integer reuse factor variable with the binary variables
        model.addConstr(reuse_factor == quicksum(rf * rf_binary_vars[rf] for rf in layer_factors))

        # Extract data for prediction
        if 'Conv1D' in layer.class_name:
            n_in = layer.get_attr('n_in') or 0
            cnn_filters = layer.get_attr('cnn_filters') or 0
            sequence_length = layer.get_attr('sequence_length') or 0

            
            X_pred = np.array([[rf, n_in, cnn_filters, sequence_length] for rf in layer_factors])
        
            latency_min_pred = quicksum(
                rf_models_conv['latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            latency_max_pred = quicksum(
                rf_models_conv['latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_conv['bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_conv['dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_conv['ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_conv['lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        elif 'LSTM' in layer.class_name:
            n_in = layer.get_attr('n_in') or 0
            sequence_length = layer.get_attr('sequence_length') or 0
            lstm_size = layer.get_attr('lstm_size') or 0

            X_pred = np.array([[rf, n_in, sequence_length, lstm_size] for rf in layer_factors])

            latency_min_pred = quicksum(
                rf_models_lstm['latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            latency_max_pred = quicksum(
                rf_models_lstm['latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_lstm['bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_lstm['dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_lstm['ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_lstm['lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        elif 'Dense' in layer.class_name:
            n_in = layer.get_attr('n_in') or 0
            n_out = layer.get_attr('n_out') or 0
            sequence_length = layer.get_attr('sequence_length') or 0

            X_pred = np.array([[rf, n_in, n_out, sequence_length] for rf in layer_factors])

            latency_min_pred = quicksum(
                rf_models_dense['latency_min'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            latency_max_pred = quicksum(
                rf_models_dense['latency_max'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            bram_usage_pred = quicksum(
                rf_models_dense['bram_18k'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            dsp_usage_pred = quicksum(
                rf_models_dense['dsp48e'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            ff_usage_pred = quicksum(
                rf_models_dense['ff'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )
            lut_usage_pred = quicksum(
                rf_models_dense['lut'].predict(X_pred[i].reshape(1, -1))[0] * rf_binary_vars[rf]
                for i, rf in enumerate(layer_factors)
            )

        # Aggregate latency and resource usage
        total_latency_min += latency_min_pred
        total_latency_max += latency_max_pred
        total_bram_usage += bram_usage_pred
        total_dsp_usage += dsp_usage_pred
        total_ff_usage += ff_usage_pred
        total_lut_usage += lut_usage_pred

    # Slack variables for big networks
    slack_bram = model.addVar(name="slack_bram", lb=0)
    slack_dsp = model.addVar(name="slack_dsp", lb=0)
    slack_ff = model.addVar(name="slack_ff", lb=0)
    slack_lut = model.addVar(name="slack_lut", lb=0)

    #constraints
    model.addConstr(total_bram_usage <= 624 * 0.95 + slack_bram, "BRAM constraint")
    model.addConstr(total_dsp_usage <= 1728 * 0.95 + slack_dsp, "DSP constraint")
    model.addConstr(total_ff_usage <= 460800 * 0.95 + slack_ff, "FF constraint")
    model.addConstr(total_lut_usage <= 230400 * 0.95 + slack_lut, "LUT constraint")

    #objective function
    model.setObjective(total_latency_min + total_latency_max + 1000 * (slack_bram + slack_dsp + slack_ff + slack_lut), GRB.MINIMIZE)
    #model.computeIIS()

    #optimize the model
    model.optimize()
    model.write("rf.sol")

    
    
    
    # warning if slack being used
    slack_used = any(var.X > 0 for var in [slack_bram, slack_dsp, slack_ff, slack_lut])
    if slack_used:
        print("WARNING: Slack variables were used, indicating that the network may not fit onto the ZCU104 Board.")
    
    # Extract optimal reuse factors 
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        for layer_name in valid_factors.keys():
            selected_rf = [rf for rf in valid_factors[layer_name] if model.getVarByName(f"rf_{layer_name}_{rf}").X > 0.5]
            if selected_rf:
                reuse_factors[layer_name] = selected_rf[0]
            else:
                print(f"No feasible solution found for layer {layer_name} that fits the resource constraints.")
    else:
        print(f"Optimizer did not find an optimal solution. Status: {model.status}")
        

    return reuse_factors







def generate_yaml_for_hls4ml(directory, filename, model_directory, optimized_reuse_factors):
    # Load the Keras model JSON
    model_json_path = os.path.join(model_directory, filename + ".json")
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

    # Assign optimized reuse factors to each layers
    for layer_name in unique_layer_names:
        # Use the optimized reusefactor
        reuse_factor = optimized_reuse_factors.get(layer_name, optimized_reuse_factors.get(unique_layer_names[0], 16))
        layer_name_config[layer_name] = {
            "ReuseFactor": reuse_factor,
            "Strategy": "Resource",
            "Compression": True
        }

    # Build YAML content
    yaml_content = {
        "Backend": "Vivado",
        "Part": "xczu7ev-ffvc1156-2-e",
        "ClockPeriod": 4,
        "IOType": "io_stream",
        "keras_json_model": os.path.join(model_directory, filename + ".json"),
        "KerasH5": os.path.join(model_directory, filename + ".h5"),
        "OutputDir": filename,
        "ProjectName": filename,
        "HLSConfig": {
            "Model": {
                "Precision": "ap_fixed<16,8,AP_RND,AP_SAT>",
                "ReuseFactor": 16,
                "Strategy": "Resource",
            },
            "LayerName": layer_name_config
        }
    }

    # Save the YAML content to a file
    yaml_path = os.path.join(directory, filename + ".yaml")
    try:
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False, sort_keys=False)
    except IOError as e:
        print(f"ERROR: Could not open {yaml_path} for writing: {e}")







# Load JSON data for the network
with open('network_112_1_23_1_57_5_83_83_83_83_83.json', 'r') as f:
    model_config = json.load(f)

# Extract layers and get valid reuse factors
analyzer = ModelAnalyzer()
layers = []

# Parse the model configuration and create Layer instances
n_lstm = sum(1 for layer_config in model_config['config']['layers'] if 'LSTM' in layer_config['class_name'])
n_cnn = sum(1 for layer_config in model_config['config']['layers'] if 'Conv1D' in layer_config['class_name'])
cnn_filters = next((layer_config['config']['filters'] for layer_config in model_config['config']['layers'] if 'Conv1D' in layer_config['class_name']), 1)

# Extracting input length from InputLayer configuration
inputs = next((layer_config['config']['batch_input_shape'][1] 
               for layer_config in model_config['config']['layers'] if layer_config['class_name'] == 'InputLayer'), 1)

# Parse the model configuration and create Layer instances
for i, layer_config in enumerate(model_config['config']['layers']):
    class_name = layer_config['class_name']
    config = layer_config['config']
    build_config = layer_config.get('build_config', {})

    if class_name == 'Dense':
        sequence_length = calculate_dense_sequence_length(i, n_lstm, n_cnn, cnn_filters, cnn_filters, inputs)
        build_config['sequence_length'] = sequence_length

    if class_name in ['Dense', 'Conv1D', 'LSTM']:
        layer = Layer(class_name, config, build_config)
        layers.append(layer)
# Analyze model to get valid reuse factors
valid_factors = analyzer.analyze_model(layers)

# Optimize reuse factors for the entire network
optimal_reuse_factors = optimize_reuse_factors_network(layers, valid_factors, rf_models_conv, rf_models_lstm, rf_models_dense)
print("Optimal Reuse Factors for Network:")
for layer_name, reuse_factor in optimal_reuse_factors.items():
    print(f"{layer_name}: {reuse_factor}")



directory="/share/ss121/hls4ml_suyash/hls4ml_explorer/training/training_new_data"
model_directory="/share/ss121/hls4ml_suyash/hls4ml_explorer/training/training_new_data"
filename="CNN_model"
#generate_yaml_for_hls4ml(directory, filename, model_directory, optimal_reuse_factors)
