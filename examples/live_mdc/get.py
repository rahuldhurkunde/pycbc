import argparse
from gwdatafind import find_urls
import os
import shutil

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Copy frame files for H and L detectors based on GPS time.")
parser.add_argument('--gps-start-time', type=int, required=True, help='Start time in GPS')
parser.add_argument('--gps-end-time', type=int, required=True, help='End time in GPS')

args = parser.parse_args()

# Define the detectors and channel
detectors = ['H', 'L', 'V']

# Get the current directory to copy files to
destination_dir = '{}/frames-without-injs'.format(os.getcwd())

# Iterate over detectors and find files
for detector in detectors:
    files = find_urls(detector, f'{detector}1_GWOSC_O3b_4KHZ_R1', args.gps_start_time, args.gps_end_time, 
                      host='datafind.gw-openscience.org')
    for file_url in files:
        # Convert the file URL to a local file path by removing 'file://localhost/'
        local_file_path = file_url.replace('file://localhost', '')
  
        # Construct the destination path
        destination_file_path = os.path.join(destination_dir, os.path.basename(local_file_path))
        
        # Copy the file to the destination directory
        shutil.copy(local_file_path, destination_file_path)
        print(f"Copied {local_file_path} to {destination_file_path}")
