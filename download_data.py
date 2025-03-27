from huggingface_hub import hf_hub_download

REPO_ID = "bkai-foundation-models/crosslingual"

import requests

# Define the repository and folders
repo_id = "bkai-foundation-models/crosslingual"
folders = ["original", "eval", "synthetic"]

# Function to list files in a folder
def list_files_in_folder(repo_id, folder):
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{folder}"
    response = requests.get(url)
    if response.status_code == 200:
        files = [file["path"] for file in response.json()]
        return files
    else:
        print(f"Failed to list files in {folder}: {response.status_code}")
        return []

# List files in each folder
all_files = {}
for folder in folders:
    files = list_files_in_folder(repo_id, folder)
    
    all_files[folder] = files
    print(f"Files in {folder}: {files}")

for folder in all_files:
    files = all_files[folder]
    for file in files:
        file = file.split("/")[1]
        hf_hub_download(repo_id=REPO_ID, subfolder=folder, repo_type="dataset", filename=file, local_dir="data")
