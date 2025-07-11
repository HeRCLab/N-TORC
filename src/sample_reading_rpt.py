#!/bin/python3

import os
import sys
import pandas as pd
import logging
import yaml

logging.basicConfig(level=logging.DEBUG)
yaml_directory = "yaml_files"

def read_rpt_file(directory):
    base_name = os.path.basename(directory)
    rpt_path = os.path.join(directory, f"{base_name}_prj", "solution1", "syn", "report", f"{base_name}_csynth.rpt")
    logging.debug(f"Reading report file: {rpt_path}")

    # Initialize these lists inside the function to reset them for each file processed.
    latency_instances = []
    latency_modules = []
    latency_min = []
    latency_max = []
    interval_min = []
    interval_max = []
    pipeline_type = []
    
    resource_instances = []
    resource_modules = []
    bram_18k = []
    dsp48e = []
    ff = []
    lut = []
    uram = []
    
    total_bram_18k = 0
    total_dsp48e = 0
    total_ff = 0
    total_lut = 0
    total_uram = 0

    fit_status = "Report file not found"
    min_latency = None
    max_latency = None

    if os.path.isfile(rpt_path):
        with open(rpt_path, "r") as file:
            lines = file.readlines()

        utilization_line = None
        for line in lines:
            if "Utilization (%)" in line:
                utilization_line = line
                break

        if utilization_line:
            parts = utilization_line.split('|')[1:-1]
            utilizations = [int(part.strip().replace('(%)', '')) for part in parts if part.strip().replace('(%)', '').isdigit()]
            fit_status = "Not Fit" if any(u > 100 for u in utilizations) else "Fit"
        else:
            fit_status = "Utilization (%) row not found."

        latency_section_found = False
        for i, line in enumerate(lines):
            if "+ Latency (clock cycles):" in line:
                latency_section_found = True
                for j in range(i+5, len(lines)):
                    latency_values_line = lines[j].strip()
                    if latency_values_line:
                        latency_values = latency_values_line.split('|')[1:-1]
                        if len(latency_values) >= 4:
                            try:
                                min_latency = int(latency_values[0].strip())
                                max_latency = int(latency_values[1].strip())
                            except ValueError as e:
                                logging.error(f"Error converting latency values to integers: {e}")
                            break
                break
        if min_latency is None or max_latency is None:
            min_latency = "Latency info not found"
            max_latency = "Latency info not found"

        # Extract detailed latency information from the first + Detail: section
        detail_section_found = False
        resource_section_counter = 0
        for i, line in enumerate(lines):
            if "+ Detail:" in line:
                resource_section_counter += 1
                detail_section_found = True
                if resource_section_counter == 1:
                    continue
            if detail_section_found and resource_section_counter == 1:
                if line.strip() == '':
                    detail_section_found = False
                    continue
                parts = line.split('|')[1:-1]
                if len(parts) == 7:
                    latency_instances.append(parts[0])
                    latency_modules.append(parts[1].strip().split('_')[0])
                    latency_min.append(int(parts[2].strip()) if parts[2].strip().isdigit() else 0)
                    latency_max.append(int(parts[3].strip()) if parts[3].strip().isdigit() else 0)
                    interval_min.append(int(parts[4].strip()) if parts[4].strip().isdigit() else 0)
                    interval_max.append(int(parts[5].strip()) if parts[5].strip().isdigit() else 0)
                    pipeline_type.append(parts[6].strip())
            if detail_section_found and resource_section_counter == 2:
                if line.strip() == '':
                    break  # End of resource section
                parts = line.split('|')[1:-1]
                if len(parts) == 7:
                    #resource_instances.append(parts[0].strip().split('_')[0])
                    resource_instances.append(parts[0])
                    resource_modules.append(parts[1].strip().split('_')[0])
                    bram_18k.append(int(parts[2].strip()) if parts[2].strip().isdigit() else 0)
                    dsp48e.append(int(parts[3].strip()) if parts[3].strip().isdigit() else 0)
                    ff.append(int(parts[4].strip()) if parts[4].strip().isdigit() else 0)
                    lut.append(int(parts[5].strip()) if parts[5].strip().isdigit() else 0)
                    uram.append(int(parts[6].strip()) if parts[6].strip().isdigit() else 0)

        # Extract total estimates
        utilization_section_found = False
        for i, line in enumerate(lines):
            if "== Utilization Estimates" in line:
                utilization_section_found = True
            if utilization_section_found and "|Total" in line:
                total_estimates_line = lines[i].strip()
                parts = total_estimates_line.split('|')[1:-1]
                if len(parts) == 6:
                    total_bram_18k = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
                    total_dsp48e = int(parts[2].strip()) if parts[2].strip().isdigit() else 0
                    total_ff = int(parts[3].strip()) if parts[3].strip().isdigit() else 0
                    total_lut = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                    total_uram = int(parts[5].strip()) if parts[5].strip().isdigit() else 0
                break

    params = base_name.split('_')[1:]
    while len(params) < 7:
        params.append(None)
    params = params[:7]  # Ensure params list has exactly 7 elements

    yaml_path = os.path.join(yaml_directory, f"{base_name}.yaml")
    reuse_factor = None
    if os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            reuse_factor = (
                yaml_content.get('HLSConfig', {}).get('Model', {}).get('ReuseFactor') or
                yaml_content.get('HLSConfig', {}).get('Model', {}).get('LayerType', {}).get('Dense', {}).get('ReuseFactor')
            )
       

    return params + [fit_status, min_latency, max_latency, reuse_factor, latency_instances, latency_modules, latency_min, latency_max, interval_min, interval_max, pipeline_type, resource_instances, resource_modules, bram_18k, dsp48e, ff, lut, uram, total_bram_18k, total_dsp48e, total_ff, total_lut, total_uram]

def get_model_cost(n_CNN_layers, CNN_filters_size, n_LSTM_layers, LSTM_units, n_MLP_layers, MLP_units, input_len):
    # CNN Layer Multiplies Calculation
    cnn_cost = 0
    input_seq_length = input_len
    output_seq_length = input_len
    num_filters = CNN_filters_size

    for i in range(n_CNN_layers):
        cnn_cost += input_seq_length * input_len * num_filters[i] * output_seq_length
        output_seq_length = output_seq_length // 2  # Assuming pooling reduces length by half in each layer
        input_len = num_filters[i]  # Output size becomes input size for next layer

    # LSTM Layer Multiplies Calculation
    lstm_cost = 0
    lstm_input_seq_length = output_seq_length  # Input length from CNN layer
    lstm_input_size = input_len  # Number of filters from CNN layer

    for i in range(n_LSTM_layers):
        lstm_cost += lstm_input_seq_length * lstm_input_size * 4 * (LSTM_units[i] ** 2)
        lstm_input_size = LSTM_units[i]  # Output from LSTM layer as input to the next layer

    # MLP (Dense) Layer Multiplies Calculation
    mlp_cost = 0
    mlp_input_size = lstm_input_size  # Output size from LSTM

    for i in range(n_MLP_layers):
        mlp_cost += mlp_input_size * MLP_units[i]
        mlp_input_size = MLP_units[i]  # Output of the current layer is input to the next layer

    # Total Cost Calculation
    total_cost = cnn_cost + lstm_cost + mlp_cost
    print('total_cost =', total_cost)
    total_cost = round(total_cost / 1000000, 4)  # Rounding to millions
    return total_cost


def calculate_operations(params):
    inputs, cnn_layers, cnn_filters, lstm_layers, lstm_size, dense_layers, dense_size = map(int, params[:7])
    total_cost = get_model_cost(
        cnn_layers,
        [cnn_filters] * cnn_layers,  
        lstm_layers,
        [lstm_size] * lstm_layers, 
        dense_layers,
        [dense_size] * dense_layers,  
        inputs
    )
    return total_cost

def calculate_flops(total_cost, min_latency, max_latency, clock_period=4):
    clock_period_s = clock_period * 1e-9
    if isinstance(min_latency, int) and isinstance(max_latency, int):
        min_flops = total_cost / (min_latency * clock_period_s)
        max_flops = total_cost / (max_latency * clock_period_s)
        min_gflops = min_flops / 1e9
        max_gflops = max_flops / 1e9
        return min_gflops, max_gflops
    else:
        return "Latency info not found. Skipping GFLOPS calculation", "Latency info not found. Skipping GFLOPS calculation"

def main(directory, output_csv):
    records = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            record = read_rpt_file(subdir_path)
            if record:
                total_cost = calculate_operations(record[:7])
                min_gflops, max_gflops = calculate_flops(total_cost, record[8], record[9])
                records.append(record + [min_gflops, max_gflops, total_cost])

    columns = [
        'inputs', 'cnn_layers', 'cnn_filters', 'lstm_layers', 'lstm_size', 'dense_layers', 'dense_size', 'fit_status', 
        'min_latency', 'max_latency', 'reuse_factor', 'latency_instance', 'latency_module', 'latency_min', 'latency_max', 
        'interval_min', 'interval_max', 'pipeline_type', 'resource_instance', 'resource_module', 'bram_18k', 'dsp48e', 'ff', 
        'lut', 'uram', 'total_bram_18k', 'total_dsp48e', 'total_ff', 'total_lut', 'total_uram', 'min_gflops(GFLOPS)', 'max_gflops(GFLOPS)', 'total_cost'
    ]

    # Ensure the number of columns matches the length of records
    df = pd.DataFrame(records, columns=columns[:len(records[0]) + 3])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_rpt.py <directory> <output_csv>")
        sys.exit(1)
    directory = sys.argv[1]
    output_csv = sys.argv[2]
    main(directory, output_csv)
