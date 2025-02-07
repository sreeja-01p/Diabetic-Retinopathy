import os
import zipfile
import torch

if torch.cuda.is_available():
    print(f"Running on CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. The script will run on CPU.")

dataset_name = "aptos2019-blindness-detection"  
extract_folder = "dataset"  

if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

os.system(f"kaggle competitions download -c {dataset_name} -p {extract_folder}")

zip_file_path = os.path.join(extract_folder, f"{dataset_name}.zip")

if os.path.exists(zip_file_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Dataset extracted to '{extract_folder}'.")

    os.remove(zip_file_path)
    print("Zip file removed after extraction.")
else:
    print("Error: Zip file not found. Check if the dataset was downloaded properly.")
