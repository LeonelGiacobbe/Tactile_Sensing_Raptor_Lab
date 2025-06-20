import os
import re

# Path to the parent directory containing the folders
parent_dir = '../alternating_dataset/empty/'

# Regular expression to match tr_x*, capturing x as a group
pattern = re.compile(r'^tr_(\d+)(.*)$')

# Collect all matching folders and their parsed components
folders_to_rename = []
for folder in os.listdir(parent_dir):
    match = pattern.match(folder)
    if match:
        x = int(match.group(1))
        rest = match.group(2)
        folders_to_rename.append((x, rest, folder))

# Sort the list in reverse order (highest x first)
folders_to_rename.sort(reverse=True)

# Perform renaming
for x, rest, folder in folders_to_rename:
    new_x = x + 15
    new_name = f"tr_{new_x}{rest}"
    old_path = os.path.join(parent_dir, folder)
    new_path = os.path.join(parent_dir, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed '{folder}' to '{new_name}'")
