import gdown
from tqdm import tqdm
import os

# Define the folder URL
url = "https://drive.google.com/drive/u/0/folders/1hiV_IaEcUtFcBq3tG81L0rQFArdJSYf9"

# Define the destination folder for downloading files
destination_folder = "../data/recording/"

# Make sure the destination folder exists, or create it
os.makedirs(destination_folder, exist_ok=True)

# Define a function to download the folder with a progress bar
def download_folder_with_progress(url, destination_folder):
    # First, gdown needs to fetch all files inside the folder
    file_list = gdown.download_folder(url, use_cookies=False)
    
    # Loop through all the downloaded files
    for file in tqdm(file_list, desc="Downloading files", unit="file"):
        print(f"Downloading {file}")
        
        # Define the full path to save the file
        output = os.path.join(destination_folder, os.path.basename(file))
        
        # Download the file to the specified folder
        gdown.download(file, output, quiet=False)

# Call the function with the specified folder
download_folder_with_progress(url, destination_folder)