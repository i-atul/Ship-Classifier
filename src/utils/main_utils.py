import os
import json
import base64
from pathlib import Path
from typing import Any, Iterable
import yaml
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from src.logger import logging
from PIL import Image
from torch.utils.data import Dataset

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its contents as a ConfigBox."""
    try:
        with path_to_yaml.open() as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file loaded: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        logging.error(f"Error reading YAML file {path_to_yaml}: {e}")
        raise

@ensure_annotations
def create_directories(paths: Iterable[Path], verbose: bool = True):
    """Create directories from a list of Path objects."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logging.info(f"Directory created: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Save a dictionary as a JSON file."""
    try:
        with path.open("w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON file saved: {path}")
    except Exception as e:
        logging.error(f"Error saving JSON file {path}: {e}")
        raise

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load a JSON file and return its contents as a ConfigBox."""
    try:
        with path.open() as f:
            content = json.load(f)
        logging.info(f"JSON file loaded: {path}")
        return ConfigBox(content)
    except Exception as e:
        logging.error(f"Error loading JSON file {path}: {e}")
        raise

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save data as a binary file using joblib."""
    try:
        joblib.dump(value=data, filename=path)
        logging.info(f"Binary file saved: {path}")
    except Exception as e:
        logging.error(f"Error saving binary file {path}: {e}")
        raise

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load data from a binary file using joblib."""
    try:
        data = joblib.load(path)
        logging.info(f"Binary file loaded: {path}")
        return data
    except Exception as e:
        logging.error(f"Error loading binary file {path}: {e}")
        raise

@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB as a string."""
    try:
        size_in_kb = round(path.stat().st_size / 1024)
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logging.error(f"Error getting file size for {path}: {e}")
        raise

def decode_image(imgstring: str, filename: Path):
    """Decode a base64 string and save as an image file."""
    try:
        imgdata = base64.b64decode(imgstring)
        with filename.open('wb') as f:
            f.write(imgdata)
        logging.info(f"Image decoded and saved: {filename}")
    except Exception as e:
        logging.error(f"Error decoding image to {filename}: {e}")
        raise

def encode_image_to_base64(image_path: Path) -> bytes:
    """Encode an image file to a base64 byte string."""
    try:
        with image_path.open("rb") as f:
            encoded = base64.b64encode(f.read())
        logging.info(f"Image encoded to base64: {image_path}")
        return encoded
    except Exception as e:
        logging.error(f"Error encoding image {image_path} to base64: {e}")
        raise

def get_valid_image_paths(directory: Path, exts={'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}):
    """
    Scan a directory recursively and return a list of valid image file paths that can be opened by PIL.
    Skips unreadable or corrupted images.
    """
    valid_paths = []
    directory = Path(directory)
    for path in directory.rglob('*'):
        if path.suffix.lower() in exts:
            try:
                with Image.open(path) as img:
                    img.verify()  
                valid_paths.append(str(path))
            except Exception as e:
                logging.warning(f"Skipping unreadable image: {path} ({e})")
    return valid_paths

class MyImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        class_names = sorted({Path(p).parent.name for p in image_paths})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_name = Path(img_path).parent.name
        label = self.class_to_idx[label_name]
        return image, label