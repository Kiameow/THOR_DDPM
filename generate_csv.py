import os
import csv

# Configuration
FOLDER_PATH = 'data/OPMED/masks'  # Target folder (relative to the script's location)
OUTPUT_CSV = 'data/OPMED/masks.csv'  # Output CSV filename

# Collect file paths with relative paths
file_paths = []
for file_name in sorted(os.listdir(FOLDER_PATH)):
    # Create full path and verify it's a file
    full_path = os.path.join(FOLDER_PATH, file_name)
    if os.path.isfile(full_path):
        # Create relative path and format with forward slashes
        relative_path = os.path.join('.', FOLDER_PATH, file_name)
        relative_path = relative_path.replace(os.sep, '/')  # Ensure forward slashes
        file_paths.append(relative_path)

# Write to CSV file
with open(OUTPUT_CSV, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['filename'])  # Write header
    for path in file_paths:
        writer.writerow([path])

print(f'Successfully exported {len(file_paths)} file paths to {OUTPUT_CSV}')
