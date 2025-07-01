import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from src.entity.config_entity import TrainingConfig
from src.utils.main_utils import get_valid_image_paths

class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename
        self.device = torch.device("cpu")
        self.config = TrainingConfig()
        self.class_to_idx = self._get_class_to_idx()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.model = self._load_model()

    def _get_class_to_idx(self):
        """
        Infers class-to-index mapping from the training data directory structure.
        Returns:
            dict: Mapping from class name to index.
        """
        image_dir = self.config.training_data
        valid_image_paths = get_valid_image_paths(image_dir)
        class_names = sorted({Path(p).parent.name for p in valid_image_paths})
        return {cls_name: idx for idx, cls_name in enumerate(class_names)}

    def _load_model(self):
        """
        Loads the trained VGG16 model with the correct classifier head.
        Returns:
            torch.nn.Module: The loaded model.
        """
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, self.config.params_classes)
        model.load_state_dict(torch.load(str(self.config.trained_model_path), map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocesses the input image for prediction.
        Args:
            image_path (str): Path to the image file.
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        preprocess = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def predict(self) -> dict:
        """
        Runs prediction on the input image and returns the predicted class label.
        Returns:
            dict: Prediction result with class label.
        """
        image_tensor = self._preprocess_image(self.filename).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_idx = predicted.item()
        prediction = self.idx_to_class.get(pred_idx, str(pred_idx))
        print(f"Predicted class: {prediction}")
        return {"image": prediction}
    

# if __name__ == "__main__":
#     image_path = Path("artifacts/data_ingestion/data_ingestion/ships/tanker/1l-image-120.jpg")
#     predictor = PredictionPipeline(str(image_path))
#     result = predictor.predict()
#     print(result) 