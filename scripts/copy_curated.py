# After using audioset.py to download recordings, then plotting them, deleting images and
# using curate.py to create a CSV of curated segments, you can use this script to copy the
# curated clips from where you downloaded them.

import argparse
import inspect
import os
import shutil
import sys
import pandas as pd

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default=None, help='Path to CSV containing curated list.')
parser.add_argument('-i', type=str, default=None, help='Input directory.')
parser.add_argument('-o', type=str, default=None, help='Output directory.')

args = parser.parse_args()

if args.c is None or args.i is None or args.o is None:
    print("Error. Arguments -c, -i and -o are all required.")
    quit()

csv_path = args.c
input_dir = args.i
output_dir = args.o

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

curated = pd.read_csv(csv_path)
count = 0
for i, row in curated.iterrows():
    youtube_id = row["YTID"]
    start_seconds = row["start_seconds"]
    filename = f"{youtube_id}-{int(start_seconds)}.mp3"
    input_path = os.path.join(input_dir, filename)
    if os.path.exists(input_path):
        output_path = os.path.join(output_dir, filename)
        shutil.copy(input_path, output_path)
        count += 1
    else:
        print(f"File not found: {input_path}")

print(f"# copied = {count}")
