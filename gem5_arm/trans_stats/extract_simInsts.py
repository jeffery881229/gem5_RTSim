import os
import re
from collections import defaultdict

# Compile regex to match directory names
pattern = re.compile(r"m5out_([^_]+(?:_[^_]+)?)_(\d+)_CPU(\d+)(_pim)?", re.IGNORECASE)

# Initialize nested dict: data[dataset][imagesize][cpu][mode] = simInsts
data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Walk through current directory
for entry in os.listdir('.'):
    match = pattern.match(entry)
    if match and os.path.isdir(entry):
        dataset_raw = match.group(1)
        imagesize = match.group(2)
        cpu = match.group(3)
        pim_flag = match.group(4)
        mode = 'PIM' if pim_flag else 'baseline'
        
        # Build path to stats.txt
        stats_path = os.path.join(entry, 'stats.txt')
        if os.path.isfile(stats_path):
            with open(stats_path, 'r') as f:
                for line in f:
                    if line.startswith('system.cpu.cpi') or line.startswith('system.cpu0.cpi'):
                        sim_value = line.split()[1]
                        break
                else:
                    sim_value = 'N/A'
        else:
            sim_value = 'N/A'
        
        dataset_key = dataset_raw.lower()
        if dataset_key == 'cifar' or dataset_key == 'CIFAR':
            dataset = 'CIFAR-10'
        elif dataset_key == '102_flower':
            dataset = 'Oxford_102_Flower'
        elif dataset_key == 'svhn' or dataset_key == 'SVHN':
            dataset = 'SVHN'
        else:
            dataset = dataset_raw
        
        data[dataset][imagesize][cpu][mode] = sim_value

# Define orderings
datasets = sorted(data.keys(), key=lambda x: x.lower())
imagesizes = ['224', '128', '64', '32']
cpus = ['2', '3']
modes = ['baseline', 'PIM']

# Prepare header rows
header1 = ["",]
for size in imagesizes:
    header1.append(f"{size} x {size}  ")
header2 = ["",]
for _ in imagesizes:
    header2.append("CPU2 CPU3 ")
header3 = ["transformer"]
for _ in imagesizes:
    header3.append("baseline PIM  baseline PIM ")

# Print headers
print(" ".join(header1))
print(" ".join(header2))
print(" ".join(header3))

# Print data rows
for dataset in datasets:
    row = [dataset]
    for size in imagesizes:
        for cpu in cpus:
            base_val = data[dataset][size].get(cpu, {}).get('baseline', 'N/A')
            pim_val = data[dataset][size].get(cpu, {}).get('PIM', 'N/A')
            row.extend([base_val, pim_val])
    print(" ".join(row))
