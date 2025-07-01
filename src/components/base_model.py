import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from src.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    """
    Prepares a VGG16 base model for transfer learning using PyTorch.
    Loads pretrained weights if specified, freezes all layers, and saves the model weights for later use.
    The classifier is NOT removed, so the saved state_dict matches the full VGG16 structure.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def get_base_model(self) -> None:
        """
        Loads the VGG16 model with or without pretrained weights, freezes all layers,
        moves the model to the appropriate device, and saves the model weights.
        The classifier is NOT removed.
        """
        # Select weights
        weights = self.config.params_weights
        if weights == "imagenet":
            vgg_weights = models.VGG16_Weights.IMAGENET1K_V1
        else:
            vgg_weights = None
        # Load model
        self.model = models.vgg16(weights=vgg_weights)
        # Move to device
        self.model = self.model.to(self.device)
        # Save model weights (before freezing, so state_dict is clean)
        self.save_model(self.config.base_model_path, self.model)
        # Freeze all layers (for transfer learning, after saving base)
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Base VGG16 model prepared and saved to {self.config.base_model_path}")

    @staticmethod
    def save_model(path: Path, model: nn.Module) -> None:
        """
        Saves the model weights (state_dict) to the specified path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(path))


if __name__ == "__main__":
    from src.entity.config_entity import PrepareBaseModelConfig
    config = PrepareBaseModelConfig()
    base_model = PrepareBaseModel(config)
    base_model.get_base_model()

