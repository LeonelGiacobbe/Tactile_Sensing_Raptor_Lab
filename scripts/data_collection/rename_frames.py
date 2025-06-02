import os
import re

root_folder = '.'  # running from 'a'

pattern = re.compile(r'([12])_.*_frame(\d+)\.jpg')

# Walk through each trial folder
for trial_name in os.listdir(root_folder):
    trial_path = os.path.join(root_folder, trial_name)
    if not os.path.isdir(trial_path):
        continue  # skip files, only folders

    # Process each subfolder inside the trial folder
    for subfolder_name in os.listdir(trial_path):
        subfolder_path = os.path.join(trial_path, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        print(f'\nProcessing: {subfolder_path}')

        # Step 1: Collect files and extract nums for this subfolder
        files = []
        nums = set()
        for filename in os.listdir(subfolder_path):
            match = pattern.match(filename)
            if match:
                prefix = match.group(1)
                num = int(match.group(2))
                files.append((filename, prefix, num))
                nums.add(num)

        if not files:
            print("No matching files found here, skipping.")
            continue

        # Step 2: Sort nums and create mapping old_num -> new_num starting at 1
        sorted_nums = sorted(nums)
        num_map = {old_num: new_num for new_num, old_num in enumerate(sorted_nums, start=1)}

        # Step 3: Rename files in this subfolder
        for filename, prefix, old_num in files:
            new_num = num_map[old_num]
            new_filename = re.sub(r'_frame\d+\.jpg', f'_frame{new_num}.jpg', filename)
            old_path = os.path.join(subfolder_path, filename)
            new_path = os.path.join(subfolder_path, new_filename)

            print(f'Renaming {filename} -> {new_filename}')
            os.rename(old_path, new_path)
