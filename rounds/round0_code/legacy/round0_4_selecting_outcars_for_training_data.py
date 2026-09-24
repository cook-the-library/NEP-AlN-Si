# /// script
# dependencies = [
#   "mp-api",
#   "numpy",
#   "ase",
# ]
# ///

# 1. Combine all exclude lists into ONE master set
# Using a set for all exclusions ensures O(1) lookup speed
to_remove = set()

# Add maxed out jobs
with open('round0_3_maxed_out_jobs.txt', 'r', encoding='utf-8') as f:
    to_remove.update(line.strip() for line in f)

# Add timed out or cancelled jobs
with open('round0_3_timed_out_or_cancelled.txt', 'r', encoding='utf-8') as f:
    to_remove.update(line.strip() for line in f)

# 2. Filter the main path list in one pass
with open('round0_1_vasp_job_paths.txt', 'r', encoding='utf-8') as f:
    # We only keep the path if it isn't in our master 'to_remove' set
    filtered_paths = [line.strip() for line in f if line.strip() not in to_remove]

# 3. Save the cleaned list
with open('round0_4_filters_job_paths.txt', 'w', encoding='utf-8') as f_out:
    f_out.write("\n".join(filtered_paths))

print(f"Filtering complete. {len(filtered_paths)} jobs remaining.")

#!pip install ase

import os
import random
from ase.io import read, write


def convert_and_split_data(path_list_file, output_folder, train_ratio=0.8):
    """
    Reads directory paths from a txt file, collects all OUTCAR frames,
    shuffles them, and splits them into train.xyz and test.xyz.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_configs = []

    # 1. Collect all frames from all OUTCARs
    with open(path_list_file, 'r') as f:
        directories = [line.strip() for line in f if line.strip()]

    for directory in directories:
        outcar_path = os.path.join(directory, "OUTCAR")
        if not os.path.exists(outcar_path):
            print(f"Skipping: OUTCAR not found in {directory}")
            continue

        try:
            # Read every frame from this OUTCAR
            frames = read(outcar_path, index=':')
            all_configs.extend(frames)
            print(f"Loaded {len(frames)} frames from {directory}")
        except Exception as e:
            print(f"Error reading {outcar_path}: {e}")

    if not all_configs:
        print("No data found. Check your paths.txt.")
        return

    # 2. Shuffle to ensure the test set is representative
    # Random seed (42) ensures you get the same split if you run it again
    random.seed(42)
    random.shuffle(all_configs)

    # 3. Calculate Split
    total_count = len(all_configs)
    train_count = int(total_count * train_ratio)
    
    train_set = all_configs[:train_count]
    test_set = all_configs[train_count:]

    # 4. Write Files
    train_path = os.path.join(output_folder, "train.xyz")
    test_path = os.path.join(output_folder, "test.xyz")

    write(train_path, train_set, format='extxyz')
    write(test_path, test_set, format='extxyz')

    print("-" * 30)
    print(f"Total frames: {total_count}")
    print(f"Saved {len(train_set)} frames to {train_path} (80%)")
    print(f"Saved {len(test_set)} frames to {test_path} (20%)")

# --- Run the Script ---
convert_and_split_data('4_filters_job_paths.txt', 'nep_data')

import os
import random
from ase.io import read, write

def convert_and_split_data_filter(path_list_file, output_folder, filter_keywords, train_ratio=0.8):
    """
    Reads directory paths from a txt file, FILTERS them if they contain 
    ANY of the keywords in filter_keywords (OR logic), then splits into train/test.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_configs = []

    # 1. Collect and FILTER directory paths using OR logic
    with open(path_list_file, 'r') as f:
        # 'any(k in line for k in filter_keywords)' returns True if 
        # at least one keyword is found in the directory path string.
        directories = [
            line.strip() for line in f 
            if line.strip() and any(k in line for k in filter_keywords)
        ]

    print(f"Filtering complete. Found {len(directories)} matching directories.")

    for directory in directories:
        outcar_path = os.path.join(directory, "OUTCAR")
        if not os.path.exists(outcar_path):
            print(f"Skipping: OUTCAR not found in {directory}")
            continue

        try:
            frames = read(outcar_path, index=':')
            all_configs.extend(frames)
            print(f"Loaded {len(frames)} frames from {directory}")
        except Exception as e:
            print(f"Error reading {outcar_path}: {e}")

    if not all_configs:
        print("No data found matching your filters. Check your paths and keywords.")
        return

    # 2. Shuffle
    random.seed(42)
    random.shuffle(all_configs)

    # 3. Split
    total_count = len(all_configs)
    train_count = int(total_count * train_ratio)
    
    train_set = all_configs[:train_count]
    test_set = all_configs[train_count:]

    # 4. Write Files
    train_path = os.path.join(output_folder, "train.xyz")
    test_path = os.path.join(output_folder, "test.xyz")

    write(train_path, train_set, format='extxyz')
    write(test_path, test_set, format='extxyz')

    print("-" * 30)
    print(f"Total frames collected: {total_count}")
    print(f"Saved {len(train_set)} frames to {train_path}")
    print(f"Saved {len(test_set)} frames to {test_path}")

# --- Run the Script ---
# Define your keywords in a list
my_filters = ['AlN_mp-661', 'AlN_mp-1700', 'AlN_mp-1330','Si_mp-149', 'Si_mp-165', 'Si_mp-1079297', 'Si_mp-1095269', 'Si_mp-168']

convert_and_split_data_filter('4_filters_job_paths.txt', 'nep_data_filtered_experimentally_existed', filter_keywords=my_filters)
