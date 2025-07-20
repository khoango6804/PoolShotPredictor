from roboflow import Roboflow
import os
from pathlib import Path

def download_dataset(api_key, workspace, project, version):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project
    project = rf.workspace(workspace).project(project)
    
    # Download dataset
    dataset = project.version(version).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location

if __name__ == "__main__":
    # Your Roboflow credentials
    API_KEY = "5CCGc7tH2zsRK2zK0eNI"  # Roboflow API key
    WORKSPACE = "khoangoo"
    PROJECT = "billiards-y0wwp-et7re-ggq8j"
    VERSION = 2 # or the version number you want to download
    
    # Create data directory
    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    dataset_path = download_dataset(API_KEY, WORKSPACE, PROJECT, VERSION)
    
    print("Dataset download completed!") 