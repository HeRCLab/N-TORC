import os
import yaml
import json
import subprocess
import multiprocessing
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM,Dense, Flatten,MaxPooling1D
from multiprocessing import Pool, Semaphore

# General search options
input_opts = [128, 256,512 ]
cnn_layers_opts = [1,2,4]
cnn_filters_opts = [16, 32]
lstm_layers_opts = [0, 1, 2]
lstm_size_opts = [8, 16, 32]
reuse_factors = [1,2,4, 16, 32,64, 128,512]
dense_layers_opts = [1, 2, 4]
dense_size_opts = [16, 32, 64]

# Directories
model_directory = "/share/ss121/hls4ml_suyash/hls4ml_explorer/model_json"
yaml_directory = "/share/ss121/hls4ml_suyash/hls4ml_explorer/yaml_files"

# Create directories if they don't exist
os.makedirs(model_directory, exist_ok=True)
os.makedirs(yaml_directory, exist_ok=True)

def generate_model(inputs, cnn_layers, cnn_filters, lstm_layers, lstm_size,dense_layer,dense_size):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.Input(shape=(inputs,1)))
    #CNN layers
    for _ in range(cnn_layers):
        model.add(tf.keras.layers.Conv1D(cnn_filters, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same'))

   
    
    #LSTM layers
    for _ in range(lstm_layers):
        model.add(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
     # Flatten before Dense layers
    model.add(tf.keras.layers.Flatten())
    
    #Dense layers
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(dense_size, activation='relu'))
     # Output layer
    model.add(tf.keras.layers.Dense(1))
    model.build()
    
    return model

def save_model_to_files(model, directory, filename):
    # JSON file
    json_filepath = os.path.join(directory, filename + ".json")
    model_json = model.to_json()
    with open(json_filepath, "w") as json_file:
        json_file.write(model_json)
    
    #HDF5 file
    h5_filepath = os.path.join(directory, filename + ".h5")
    model.save(h5_filepath)

def generate_yaml_for_hls4ml(directory, filename, inputs, cnn_layers, cnn_filters, lstm_layers, lstm_size,dense_layers,dense_size, reuse_factor):
    strategy= "Latency" if reuse_factor==1 else "Resource"
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
                "ReuseFactor": reuse_factor,
                "Strategy": strategy,         
               
            }
        }
    }
    
    filepath = os.path.join(directory, filename + ".yaml")
    try:
        with open(filepath, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    except IOError as e:
        print(f"ERROR: Could not open {filepath}: {e}")

def convert_yaml_to_hls4ml(yaml_file):
    command = f"hls4ml convert -c {yaml_file}"
    subprocess.run(command, shell=True, check=True)



# Generate and save models for all combinations of options
for inputs in input_opts:
    for cnn_layers in cnn_layers_opts:
        for cnn_filters in cnn_filters_opts:
            for lstm_layers in lstm_layers_opts:
                for lstm_size in lstm_size_opts:
                    for dense_layers in dense_layers_opts:
                        for dense_size in dense_size_opts:
                            for reuse_factor in reuse_factors:
                                model = generate_model(inputs, cnn_layers, cnn_filters, lstm_layers, lstm_size,dense_layers,dense_size)
                                filename = f"network_{inputs}_{cnn_layers}_{cnn_filters}_{lstm_layers}_{lstm_size}_{dense_layers}_{dense_size}_{reuse_factor}"
                                save_model_to_files(model, model_directory, filename)
                                generate_yaml_for_hls4ml(yaml_directory, filename, inputs, cnn_layers, cnn_filters, lstm_layers, lstm_size,dense_layers,dense_size,reuse_factor)
                                


# Specify the yaml directory
yaml_directory = "/share/ss121/hls4ml_suyash/hls4ml_explorer/yaml_files"
os.chdir(yaml_directory)  

yaml_files = [f for f in os.listdir(yaml_directory) if f.endswith(".yaml")]
def convert_yaml_to_hls(yaml_file):
     command=f"hls4ml convert -c {yaml_file}"
     subprocess.run(command,shell=True)
yaml_files = [os.path.join(yaml_directory, f) for f in os.listdir(yaml_directory) if f.endswith(".yaml")]
for yaml_file in yaml_files:
    convert_yaml_to_hls4ml(yaml_file)


def run_hls4ml_build(directory):
    #command=f"hls4ml build -p {directory} -a"
    command =f"cd {directory} && vivado_hls -f build_prj.tcl \"csim=1 synth=1 \""
    os.system(command)
yaml_directory = "/share/ss121/hls4ml_suyash/hls4ml_explorer/yaml_files"

directories = [f for f in os.listdir(yaml_directory) if os.path.isdir(os.path.join(yaml_directory, f))]







   # run_hls4ml_build(os.path.join(yaml_directory, directory))
def run_builds(directory):
    run_hls4ml_build(os.path.join(yaml_directory, directory))

#num_processes=cpu_count() // 400
#with Pool(processes=num_processes) as pool:
    #pool.map(convert_yaml_to_hls,yaml_files)
     #pool.map(run_builds, directories)
def process_directory(directory):
    with pool:
        run_hls4ml_build(directory)

pool = multiprocessing.Semaphore(75)  # Limit to 75 concurrent processes
with multiprocessing.Pool() as pool:
    pool.map(process_directory, directories)


