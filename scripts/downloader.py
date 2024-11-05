
import os
import zipfile
import shutil

# Define download links for the datasets
download_links = {
    "dataset1": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t2r6rszp5c-1.zip",
    "dataset2": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tgv3zb82nd-1.zip"
}

# Directory paths
download_directory = "datasets"
combined_directory = os.path.join("..", "data", "ImageData")  # Save in a "data" directory, at the same level as "scripts"

# Create directories if they don't exist
os.makedirs(download_directory, exist_ok=True)
os.makedirs(combined_directory, exist_ok=True)

def download_file(url, save_path):
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful
    file_size = int(response.headers.get("Content-Length", 0))

    with open(save_path, "wb") as file:
        print(f"Downloading {os.path.basename(save_path)} ({file_size / (1024 * 1024):.2f} MB)...")
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
        print(f"Download complete: {save_path}")

def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {file_path} to {extract_to}")

        # Check for nested ZIP files and extract them
        for root, _, files in os.walk(extract_to):
            for file in files:
                if file.endswith('.zip'):
                    nested_zip_path = os.path.join(root, file)
                    nested_extract_to = os.path.join(root, os.path.splitext(file)[0])
                    os.makedirs(nested_extract_to, exist_ok=True)
                    extract_zip(nested_zip_path, nested_extract_to)
                    os.remove(nested_zip_path)  # Optionally delete the nested ZIP after extraction

# Download and extract each dataset
for name, url in download_links.items():
    zip_path = os.path.join(download_directory, f"{name}.zip")
    try:
        download_file(url, zip_path)
        extract_zip(zip_path, combined_directory)
    except Exception as e:
        print(f"An error occurred with {name}: {e}")

        # Delete the download directory after extraction
shutil.rmtree(download_directory)
print(f"Deleted download directory: {download_directory}")

print("All datasets downloaded, extracted, and combined into:", combined_directory)