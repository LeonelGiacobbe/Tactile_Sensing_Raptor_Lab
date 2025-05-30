import os
import numpy as np

data_path = "../data_collection/dataset"

for material_dir in os.listdir(data_path):
    if not os.path.isdir(os.path.join(data_path, material_dir)) or material_dir == 'empty':
        continue
        
    label_dict = {}
    
    for trial_dir in os.listdir(os.path.join(data_path, material_dir)):
        try:
            parts = trial_dir.split('_')
            trial_num = parts[1]  # Get "25" from "tr_25_..."
            
            # Store just the gpown value as the label (or modify as needed)
            gpown = float(parts[parts.index('gpown') + 1])
            label_dict[trial_num] = gpown  # Single value instead of dict
            
        except (ValueError, IndexError):
            continue
    
    np.save(os.path.join(data_path, f"{material_dir}.npy"), label_dict)