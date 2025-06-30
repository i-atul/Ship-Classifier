import torch
from pathlib import Path
from src.entity.config_entity import EvaluationConfig
from src.utils.main_utils import get_valid_image_paths, MyImageDataset, save_json
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import mlflow
import dagshub


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        dagshub.init(repo_owner=self.config.dagshub_repo_owner, repo_name=self.config.dagshub_repo_name, mlflow=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.valid_loader = None
        self.score = None

    def load_model(self):
        # Load VGG16 architecture
        self.model = models.vgg16(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.config.params_classes)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(str(self.config.path_of_model), map_location=self.device))

    def valid_loader_setup(self):
        valid_image_paths = get_valid_image_paths(self.config.training_data)
        valid_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
        ])
        valid_dataset = MyImageDataset(valid_image_paths, transform=valid_transforms)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    def evaluate(self):
        self.load_model()
        self.valid_loader_setup()
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.valid_loader, desc="Evaluating", unit="batch"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = val_loss / total
        accuracy = correct / total
        self.score = (avg_loss, accuracy)
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return self.score

    def save_score(self):
        if self.score is None:
            raise ValueError("No evaluation score found. Run evaluate() first.")
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_to_mlflow(self):
        if self.score is None:
            raise ValueError("No evaluation score found. Run evaluate() first.")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run():
            for k, v in self.config.all_params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric('loss', self.score[0])
            mlflow.log_metric('accuracy', self.score[1])


# if __name__ == "__main__":
#     from src.entity.config_entity import EvaluationConfig
#     config = EvaluationConfig()
#     evaluator = Evaluation(config)
#     evaluator.evaluate()
#     evaluator.save_score()
#     evaluator.log_to_mlflow()

