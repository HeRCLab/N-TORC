import pandas as pd
import numpy as np
import time
import random

# Start timer
start_time = time.time()

# Load the CSV files
csv1 = pd.read_csv('conv_net2.csv')
csv2=pd.read_csv('lstm_net2.csv')
csv3 = pd.read_csv('dense_net2.csv')


# Combine all CSVs into one DataFrame
combined_df = pd.concat([csv1,csv2, csv3])

# List of columns for predicted values
predicted_columns = ['predicted_lut', 'predicted_bram', 'predicted_dsp', 'predicted_ff', 'predicted_latency_max']

# Latency constraint
latency_limit = 50000

# Function to randomly pick one reuse factor for each unique layer name
def pick_random_reuse_factors(df):
    chosen_factors = df.groupby('layer_name').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
    return chosen_factors

# Objective function to calculate total resources and check latency constraint
def calculate_objective(chosen_factors):
    aggregated_values = {col: chosen_factors[col].sum() for col in predicted_columns}

    total_latency_max = aggregated_values['predicted_latency_max']
    if total_latency_max > latency_limit:
        return float('inf'), None  # Penalize if latency exceeds the limit

    total_resources = (
        aggregated_values['predicted_bram'] +
        aggregated_values['predicted_dsp'] +
        aggregated_values['predicted_lut'] +
        aggregated_values['predicted_ff']
    )

    return total_resources, total_latency_max

# Stochastic search function
def stochastic_search(df, iterations=1000):
    best_factors = pick_random_reuse_factors(df)
    best_resources, best_latency = calculate_objective(best_factors)

    for i in range(iterations):
        # Generate a new candidate by randomly picking reuse factors
        new_factors = pick_random_reuse_factors(df)
        new_resources, new_latency = calculate_objective(new_factors)

        # Update the best solution if the new candidate is better
        if new_resources < best_resources:
            best_factors = new_factors
            best_resources = new_resources
            best_latency = new_latency

    return best_factors, best_resources, best_latency

# Run stochastic search
best_factors, best_resources, best_latency = stochastic_search(combined_df)

# Extract the optimized reuse factors
optimized_reuse_factors = ', '.join(f"{row['layer_name']}-{row['reuse_factor']}" for _, row in best_factors.iterrows())

# Save the results
results_df = pd.DataFrame({
    'network_name': ['optimized_network'],
    'total_predicted_lut': [best_factors['predicted_lut'].sum()],
    'total_predicted_bram': [best_factors['predicted_bram'].sum()],
    'total_predicted_dsp': [best_factors['predicted_dsp'].sum()],
    'total_predicted_ff': [best_factors['predicted_ff'].sum()],
    'total_predicted_latency_max': [best_latency],
    'total_resources': [best_resources],
    'optimized_reuse_factor': [optimized_reuse_factors]
})

results_df.to_csv('optimized_networks_1k.csv', index=False)
# End timer and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Process completed in {elapsed_time:.2f} seconds.")
# Save printed output to a .txt file
output_file = 'stochastic_search_results_1k.txt'
with open(output_file, 'w') as f:
    f.write(f"Process completed in {elapsed_time:.2f} seconds.\n")
    f.write(f"Optimized reuse factors: {optimized_reuse_factors}\n")
    f.write(f"Total predicted LUT: {best_factors['predicted_lut'].sum()}\n")
    f.write(f"Total predicted BRAM: {best_factors['predicted_bram'].sum()}\n")
    f.write(f"Total predicted DSP: {best_factors['predicted_dsp'].sum()}\n")
    f.write(f"Total predicted FF: {best_factors['predicted_ff'].sum()}\n")
    f.write(f"Total predicted latency max: {best_latency}\n")



# Print the results
print(f"Optimized reuse factors: {optimized_reuse_factors}")
print(f"Total predicted LUT: {best_factors['predicted_lut'].sum()}")
print(f"Total predicted BRAM: {best_factors['predicted_bram'].sum()}")
print(f"Total predicted DSP: {best_factors['predicted_dsp'].sum()}")
print(f"Total predicted FF: {best_factors['predicted_ff'].sum()}")
print(f"Total predicted latency max: {best_latency}")


