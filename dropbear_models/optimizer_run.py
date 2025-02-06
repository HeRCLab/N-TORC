import os
import csv
import subprocess

def extract_optimizer_output(output):
    """Parse the output of optimizer_update.py to extract relevant information."""
    lines = output.splitlines()
    data = {
        "predicted_latency": None,
        "predicted_lut": None,
        "predicted_bram": None,
        "predicted_dsp": None,
        "predicted_ff": None,
        "reuse_factors": "",
    }
    
    for line in lines:
        if line.startswith("Total predicted_latency_max:"):
            data["predicted_latency"] = line.split(":")[1].strip()
        elif line.startswith("Total predicted_lut:"):
            data["predicted_lut"] = line.split(":")[1].strip()
        elif line.startswith("Total predicted_bram:"):
            data["predicted_bram"] = line.split(":")[1].strip()
        elif line.startswith("Total predicted_dsp:"):
            data["predicted_dsp"] = line.split(":")[1].strip()
        elif line.startswith("Total predicted_ff:"):
            data["predicted_ff"] = line.split(":")[1].strip()
        elif line.startswith("Optimal Reuse Factors for Network:"):
            factors = []
            for factor_line in lines[lines.index(line) + 1:]:
                if ":" in factor_line:
                    factors.append(factor_line.strip())
                else:
                    break
            data["reuse_factors"] = ", ".join(factors)
    return data

def run_optimizer_and_update_csv(output_csv):
    """Run optimizer_update.py for all JSON files in the current directory and update the CSV."""
    directory = os.getcwd()  # Set directory to the current working directory
    
    # Full path to optimizer_update.py
    optimizer_script = os.path.join(directory, "optimizer_updated.py")

    if not os.path.exists(optimizer_script):
        raise FileNotFoundError(f"The optimizer script '{optimizer_script}' does not exist.")

    # List all JSON files in the directory
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")]

    if not json_files:
        print("No JSON files found in the directory.")
        return

    results = []
    for json_file in json_files:
        json_path = os.path.join(directory, json_file)
        print(f"Processing file: {json_path}")
        try:
            result = subprocess.run(
                ["python", optimizer_script, json_path],
                capture_output=True,
                text=True,
                check=True,
            )
            optimizer_data = extract_optimizer_output(result.stdout)
            optimizer_data["network_name"] = json_file
            results.append(optimizer_data)
        except subprocess.CalledProcessError as e:
            print(f"Error running optimizer for '{json_file}': {e}")
            print(f"Error message: {e.stderr}")
            continue

    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["network_name", "predicted_lut", "predicted_DSPs", 
                             "predicted_latency", "RF for each Layer"])
        for result in results:
            csv_writer.writerow([
                result["network_name"], 
                result["predicted_lut"], 
                result["predicted_dsp"], 
                result["predicted_latency"], 
                result["reuse_factors"],
            ])

    print(f"CSV file '{output_csv}' updated with results for {len(results)} networks.")

# Output CSV file
output_csv = "networks_fccm.csv"

run_optimizer_and_update_csv(output_csv)

