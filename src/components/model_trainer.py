import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm  
from src.entity.config_entity import TrainingConfig
from src.utils.main_utils import get_valid_image_paths, MyImageDataset
from sklearn.model_selection import train_test_split


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def get_base_model(self):
        # Load VGG16 architecture
        self.model = models.vgg16(weights=None)
        # Load base weights first (with original classifier)
        self.model.load_state_dict(torch.load(str(self.config.updated_base_model_path), map_location=self.device))
        # Then replace classifier for your number of classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.config.params_classes)
        self.model = self.model.to(self.device)

    def train_valid_loader(self):
        # Ensure training data directory exists
        if not os.path.exists(self.config.training_data):
            raise FileNotFoundError(f"Training data directory not found: {self.config.training_data}")

        image_dir = self.config.training_data
        valid_image_paths = get_valid_image_paths(image_dir)

        # Data augmentation and normalization for training
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.config.params_image_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) if self.config.params_is_augmentation else transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
        ])

        # Split valid_image_paths into train/val
        train_paths, val_paths = train_test_split(valid_image_paths, test_size=0.2, random_state=42)

        self.train_loader = DataLoader(MyImageDataset(train_paths, transform=train_transforms), batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(MyImageDataset(val_paths, transform=valid_transforms), batch_size=self.config.params_batch_size, shuffle=False)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)
        epochs = self.config.params_epochs

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            train_loader_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
            for images, labels in train_loader_iter:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                train_loader_iter.set_postfix(loss=loss.item())
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}')

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            valid_loader_iter = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", unit="batch")
            with torch.no_grad():
                for images, labels in valid_loader_iter:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    valid_loader_iter.set_postfix(loss=loss.item())
            val_loss /= len(self.valid_loader.dataset)
            val_acc = correct / len(self.valid_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

        self.save_model(self.config.trained_model_path, self.model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(path))


if __name__ == "__main__":
    from src.entity.config_entity import TrainingConfig
    config = TrainingConfig()
    trainer = Training(config)
    trainer.get_base_model()
    trainer.train_valid_loader()
    trainer.train()
