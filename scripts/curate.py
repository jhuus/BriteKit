# To create or update a list of curated segments:
#
# 1. Download the recordings using audioset.py
# 2. Plot them using plot_recordings.py --all
# 3. Delete any spectrograms you wish to exclude
# 4. Run this script
#

import argparse
import glob
import logging
import os
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default=None, help='Name of directory containing spectrograms.')
parser.add_argument('-c', type=str, default=None, help='Path to CSV to create or update.')
args = parser.parse_args()

input_dir, csv_path = args.i, args.c

if input_dir is None or csv_path is None:
    logging.error("Error: both -i and -c must be specified.")
    quit()

image_paths = glob.glob(os.path.join(input_dir, "*.jpeg"))
youtube_id = []
start_seconds = []
for image_path in image_paths:
    stem = Path(image_path).stem
    index = stem.rfind('-')
    if index == -1:
        logging.error(f"Error: invalid image name format = {Path(image_path).name}")
        quit()

    youtube_id.append(stem[:index])
    start_seconds.append(float(stem[index+1:]))

df = pd.DataFrame()
df["YTID"] = youtube_id
df["start_seconds"] = start_seconds

if os.path.exists(csv_path):
    # append the new rows to the existing rows
    old_df = pd.read_csv(csv_path, dtype={"start_seconds": float})
    df = pd.concat([old_df, df], ignore_index=True)

df_sorted = df.sort_values(by=["YTID", "start_seconds"])
df_unique = df_sorted.drop_duplicates()
df_unique.to_csv(csv_path, index=False)
