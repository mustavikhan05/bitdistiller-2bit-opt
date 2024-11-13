import json
import random

# Store all outputs in a list
all_outputs = []

# List of input JSON file paths
json_paths = [
    "/Users/mustavikhan/Google Drive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-125m/code_T0.7_N1024_S42_3000.json",
    "/Users/mustavikhan/Google Drive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-125m/math_T0.7_N1024_S42_3000.json",
    "/Users/mustavikhan/Google Drive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-125m/openorca-solar_T0.7_N1024_S42_3000.json"
]

# Read and combine all files
for json_path in json_paths:
    try:
        with open(json_path, 'r') as f:
            dataset_for_eval = f.readlines()
        for line in dataset_for_eval:
            json_data = json.loads(line)
            all_outputs.append(json_data)
        print(f"Successfully read: {json_path}")
    except Exception as e:
        print(f"Error reading {json_path}: {e}")

# Shuffle the combined data
random.seed(42)  # For reproducibility
random.shuffle(all_outputs)

# Write the combined and shuffled data to output file
output_path = '/Users/mustavikhan/Google Drive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-125m/mix_code_math_orca_9000.json'
try:
    with open(output_path, 'w') as f:
        for item in all_outputs:
            f.write(json.dumps(item) + '\n')
    print(f"Successfully wrote combined dataset to: {output_path}")
except Exception as e:
    print(f"Error writing output file: {e}")

print(f"Total number of examples in combined dataset: {len(all_outputs)}")