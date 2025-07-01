import os
import zipfile
import gdown
from src.logger import logging 
from src.entity.config_entity import DataIngestionConfig
from src.utils.main_utils import get_valid_image_paths
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    # Download data from the url
    def download_file(self)-> str:
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            gdrive_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+gdrive_id, zip_download_dir)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    # Extracts the zip file into the data directory
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        # Get all valid image paths
        image_dir = Path("artifacts/data_ingestion/data_ingestion")
        valid_image_paths = get_valid_image_paths(image_dir)

        # Define a default transform (resize to 224x224 and convert to tensor)
        from torchvision import transforms
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        class MyImageDataset(Dataset):
            def __init__(self, image_paths, transform=None):
                self.image_paths = image_paths
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image

       
        dataset = MyImageDataset(valid_image_paths, transform=default_transform)

if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    ingestion.download_file()
    ingestion.extract_zip_file()

