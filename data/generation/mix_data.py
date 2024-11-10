import json
import random

# Store all outputs in a list
all_outputs = []

# List of input JSON file paths
json_paths = [
    "/content/drive/MyDrive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-350m/code_T0.7_N1024_S42_3000.json",
    "/content/drive/MyDrive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-350m/math_T0.7_N1024_S42_3000.json",
    "/content/drive/MyDrive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-350m/openorca-solar_T0.7_N1024_S42_3000.json"
]

# Read and combine all files
for json_path in json_paths:
    with open(json_path, 'r') as f:
        dataset_for_eval = f.readlines()
    for line in dataset_for_eval:
        json_data = json.loads(line)
        all_outputs.append(json_data)

# Shuffle the combined data
random.shuffle(all_outputs)

# Write the combined and shuffled data to output file
output_path = '/content/drive/MyDrive/Bitdistiller-OPT-Quant/data/datasets/hf-opt-350m/mix_code_math_orca_9000.json'
with open(output_path, 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')

print(f"Total number of examples in combined dataset: {len(all_outputs)}")