import os
import re

# Path to the parent directory containing the folders
parent_dir = '.'

# Regular expression to match tr_x*, capturing x as a group
pattern = re.compile(r'^tr_(\d+)(.*)$')

for folder in os.listdir(parent_dir):
    match = pattern.match(folder)
    if match:
        x = int(match.group(1))
        rest = match.group(2)
        if x >= 48:
            new_x = x - 7
            new_name = f"tr_{new_x}{rest}"
            old_path = os.path.join(parent_dir, folder)
            new_path = os.path.join(parent_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed '{folder}' to '{new_name}'")
